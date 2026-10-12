import asyncio
import json

from goose import Agent, FlowArguments, Result, flow, task
from goose.agent import AIModel, SystemMessage, TextMessagePart, UserMessage

from token_matcher import MatchResult, TokenMatcher, load_weights, plot_distributions

OPENAI_MODEL = AIModel.GPT_4O 
ADVISOR_PATH = "advisor_answers.json"
LABELS_PATH = "token_labels.json"


class CareerGoals(FlowArguments):
    goals: list[str]
    current_stage: str
    target_role: str
    target_company: str
    target_timeline: str  # e.g., "15 years"
    model: AIModel  # which provider's agent runs this flow


class CareerStep(Result):
    step_name: str
    details: str
    reason: str


@task
async def plan_high_school(*, target_role: str, model: AIModel, agent: Agent) -> CareerStep:
    return await agent(
        messages=[
            UserMessage(parts=[
                TextMessagePart(
                    text=f"I am currently in high school and I want to become a {target_role}."
                         " What are the best steps I should take in high school to prepare for this career?"
                         " Include recommended courses, extracurriculars, and networking opportunities."
                )
            ])
        ],
        system=SystemMessage(parts=[
            TextMessagePart(
                text="You are a career advisor. Provide a structured action plan for high school students "
                     "based on their chosen career path. Do NOT assume any specific field—tailor the response dynamically "
                     "based on the user's target career."
            )
        ]),
        response_model=CareerStep,
        model=model,
        task_name="high_school"
    )


@task
async def plan_college(*, target_role: str, high_school: CareerStep, model: AIModel, agent: Agent) -> CareerStep:
    return await agent(
        messages=[
            UserMessage(parts=[
                TextMessagePart(
                    text=f"I have completed high school with the following steps:\n{high_school.details}\n"
                         f"Now, what should I do in college to advance towards becoming a {target_role}?"
                         " Include relevant degree choices, certifications, internships, and professional development."
                )
            ])
        ],
        system=SystemMessage(parts=[
            TextMessagePart(
                text="You are a career advisor. Provide a structured plan for college students pursuing any career. "
                     "Do not assume the user is in a specific field—adjust the response based on their target career."
            )
        ]),
        response_model=CareerStep,
        model=model,
        task_name="college"
    )


@task
async def plan_entry_level(*, target_role: str, college: CareerStep, model: AIModel, agent: Agent) -> CareerStep:
    return await agent(
        messages=[
            UserMessage(parts=[
                TextMessagePart(
                    text=f"I have completed college with the following steps:\n{college.details}\n"
                         f"What should I do in my entry-level job to set myself up for a successful career as a {target_role}?"
                         " Include job roles, skills to develop, and networking strategies."
                )
            ])
        ],
        system=SystemMessage(parts=[
            TextMessagePart(
                text="You are a career advisor. Provide a structured roadmap for individuals in their first job, "
                     "tailored to their target career. Recommend key skills, networking strategies, and growth opportunities."
            )
        ]),
        response_model=CareerStep,
        model=model,
        task_name="entry_level"
    )


@task
async def plan_mid_career(*, target_role: str, entry_level: CareerStep, model: AIModel, agent: Agent) -> CareerStep:
    return await agent(
        messages=[
            UserMessage(parts=[
                TextMessagePart(
                    text=f"I have gained experience in an entry-level job with the following steps:\n{entry_level.details}\n"
                         f"What are the key steps I should take at the mid-career level to advance towards becoming a {target_role}?"
                         " Include specialization areas, leadership opportunities, and key career moves."
                )
            ])
        ],
        system=SystemMessage(parts=[
            TextMessagePart(
                text="You are a career strategist. Provide a structured plan for professionals at the mid-career level, "
                     "tailoring recommendations to their chosen career path."
            )
        ]),
        response_model=CareerStep,
        model=model,
        task_name="mid_career"
    )


@task
async def plan_management(*, target_role: str, mid_career: CareerStep, model: AIModel, agent: Agent) -> CareerStep:
    return await agent(
        messages=[
            UserMessage(parts=[
                TextMessagePart(
                    text=f"I have progressed through my career with the following steps:\n{mid_career.details}\n"
                         f"What are the final steps I need to take to transition into a leadership or management role in {target_role}?"
                         " Should I pursue further education, leadership training, or networking opportunities?"
                )
            ])
        ],
        system=SystemMessage(parts=[
            TextMessagePart(
                text="You are a leadership and career strategist. Provide a structured action plan for transitioning into "
                     "a leadership or executive role, tailored to the user's career path."
            )
        ]),
        response_model=CareerStep,
        model=model,
        task_name="management"
    )


@flow
async def career_growth_plan(*, flow_arguments: CareerGoals, agent: Agent) -> None:
    high_school = await plan_high_school(target_role=flow_arguments.target_role, model=flow_arguments.model, agent=agent)
    college = await plan_college(target_role=flow_arguments.target_role, high_school=high_school, model=flow_arguments.model, agent=agent)
    entry_level = await plan_entry_level(target_role=flow_arguments.target_role, college=college, model=flow_arguments.model, agent=agent)
    mid_career = await plan_mid_career(target_role=flow_arguments.target_role, entry_level=entry_level, model=flow_arguments.model, agent=agent)
    management = await plan_management(target_role=flow_arguments.target_role, mid_career=mid_career, model=flow_arguments.model, agent=agent)

async def main() -> None:
    print("\nCareer Growth Roadmap Generator")
    current_stage = input("Enter your current stage (e.g., High School, College, Entry-Level Job): ").strip()
    target_role = input("Enter your target role: ").strip()
    target_company = input("Enter your target company (e.g., Kaiser, Google, Tesla): ").strip()
    target_timeline = input("In how many years do you want to reach this role? (e.g., 20 years): ").strip()

    print("\nEnter your career goals one at a time.")
    print("Press Enter without typing anything when you're done.\n")

    goals = []
    while True:
        goal = input("Enter a career goal: ").strip()
        if not goal:
            break
        goals.append(goal)

    # Run the identical 5-stage workflow once per provider, then score each stage
    # against the advisor's answers with the weighted token matcher.
    providers = {"gemini": AIModel.GEMINI_FLASH, "openai": OPENAI_MODEL}
    stage_tasks = {
        "high_school": plan_high_school, "college": plan_college, "entry_level": plan_entry_level,
        "mid_career": plan_mid_career, "management": plan_management,
    }
    outputs: dict[str, dict[str, str]] = {}  # provider -> stage -> text

    for label, model in providers.items():
        career_goals = CareerGoals(
            goals=goals,
            current_stage=current_stage,
            target_role=target_role,
            target_company=target_company,
            target_timeline=target_timeline,
            model=model,
        )
        async with career_growth_plan.start_run(run_id=f"career_roadmap_{label}") as run:
            await career_growth_plan.generate(career_goals)
            outputs[label] = {s: run.get(task=t).result.details for s, t in stage_tasks.items()}

    with open(ADVISOR_PATH) as f:
        advisor = json.load(f)  # {"high_school": ["line", "line", ...], ...}
    # Each answer is stored as a list of lines for readability; a plain string also works.
    advisor = {k: "\n".join(v) if isinstance(v, list) else v for k, v in advisor.items()}

    matcher = TokenMatcher()
    results: dict[str, dict[str, MatchResult]] = {}
    print("\n--- Similarity to advisor (weighted token-match F1) ---")
    for stage in stage_tasks:
        results[stage] = {}
        for label in providers:
            r = matcher.score(
                advisor[stage],
                outputs[label][stage],
                ref_weights=load_weights(LABELS_PATH, stage, "advisor"),
                cand_weights=load_weights(LABELS_PATH, stage, label),
            )
            results[stage][label] = r
            print(f"{stage:12s} {label:7s} P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}")

    plot_distributions(results, "similarity_distributions.png")
    with open("model_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)
    print("\nSaved similarity_distributions.png and model_outputs.json")

if __name__ == "__main__":
    asyncio.run(main())
