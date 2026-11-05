import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from groq import Groq


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Digital Health AI Research Platform",
    page_icon="🧠",
    layout="wide"
)

MODEL = "openai/gpt-oss-20b"
PROJECTS_FILE = "research_projects.json"

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


MODULES = [
    ("qualitative", "📝 Qualitative Analysis"),
    ("quantitative", "📊 Quantitative Analysis"),
    ("usability", "🔍 Usability / Heuristic Evaluation"),
    ("journey", "🧭 User Journey Mapping"),
    ("mixed_methods", "🔀 Mixed-Methods Synthesis"),
    ("co_design", "🤝 Co-Design"),
    ("evidence_audit", "✅ Evidence Verification / Audit"),
    ("recommendations", "💡 Evidence-Based Recommendations"),
]


# ============================================================
# STORAGE AND PROJECT MANAGEMENT
# ============================================================

def load_projects():
    if not os.path.exists(PROJECTS_FILE):
        return []

    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (OSError, json.JSONDecodeError):
        return []


def save_projects(projects):
    with open(PROJECTS_FILE, "w", encoding="utf-8") as file:
        json.dump(projects, file, indent=2, ensure_ascii=False)


def blank_project(**values):

    project = {
        "project_name": "",
        "innovation_name": "",
        "innovation_description": "",
        "target_users": "",
        "research_objective": "",
        "research_questions": "",
        "evaluation_type": "Mixed-Methods Evaluation",

        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),

        "qualitative_data": "",
        "qualitative_result": "",
        "quantitative_result": "",
        "usability_result": "",
        "journey_result": "",
        "mixed_methods_result": "",
        "co_design_brief": "",
        "co_design_result": "",
        "evidence_audit_result": "",
        "recommendations_result": "",
    }

    for key, _ in MODULES:
        project[key] = False

    project.update(values)

    return project


def seed_demo_project():

    if os.path.exists(PROJECTS_FILE):
        return

    demo = blank_project(
        project_name="ChildHealth Connect Evaluation",
        innovation_name="ChildHealth Connect",

        innovation_description=(
            "A digital health platform designed to help parents and "
            "caregivers access understandable child-health information, "
            "reminders, progress tracking, and opportunities to provide feedback."
        ),

        target_users="Parents and caregivers of young children",

        research_objective=(
            "To evaluate usability, user experience, acceptability, trust, "
            "information needs, and barriers to engagement with ChildHealth Connect."
        ),

        research_questions=(
            "1. How do parents experience using ChildHealth Connect?\n"
            "2. What barriers and facilitators affect use?\n"
            "3. How do usability problems relate to satisfaction and task completion?\n"
            "4. What design improvements should be prioritized?"
        ),
    )

    save_projects([demo])


seed_demo_project()


def normalize_project(project):

    defaults = blank_project()
    changed = False

    for key, value in defaults.items():

        if key not in project:
            project[key] = value
            changed = True

    return project, changed


def get_project(index):

    projects = load_projects()

    if not projects:
        return None, None

    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    index = max(0, min(index, len(projects) - 1))

    project, changed = normalize_project(projects[index])

    if changed:
        projects[index] = project
        save_projects(projects)

    return project, index


def update_project(index, **updates):

    project, index = get_project(index)

    if project is None:
        return None

    projects = load_projects()

    project.update(updates)

    project["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    projects[index] = project

    save_projects(projects)

    return project


def module_status(project, key):

    return "🟢 Complete" if project.get(key) else "🟡 Not completed"


def workspace_markdown(project):

    if not project:
        return "## No project selected."

    rows = "\n".join(
        f"| {label} | {module_status(project, key)} |"
        for key, label in MODULES
    )

    return f"""
# 🧠 {project['project_name']}

## Digital Health Research Project

### 🏥 Digital Health Innovation
**{project['innovation_name']}**

### 📋 Innovation Description
{project['innovation_description'] or 'Not provided'}

### 👥 Target Users
{project['target_users'] or 'Not provided'}

### 🎯 Research Objective
{project['research_objective'] or 'Not provided'}

### ❓ Research Questions
{project['research_questions'] or 'Not provided'}

### 🔬 Evaluation Type
**{project['evaluation_type']}**

---

## Research Workflow

Project Setup → Qualitative → Quantitative → Usability →
Journey Mapping → Mixed-Methods → Co-Design →
Evidence Verification/Audit → Evidence-Based Recommendations

| Module | Status |
|---|---|
{rows}
"""


# ============================================================
# GROQ AI HELPERS
# ============================================================

def get_client():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets."
        )

    return Groq(api_key=GROQ_API_KEY)


def call_ai(prompt, max_tokens=3000):

    response = get_client().chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful digital health research assistant. "
                    "Never fabricate evidence, quotations, participants, "
                    "statistics, citations, or causal claims."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def compact(text, limit):

    return (text or "").strip()[:limit]


def ai_error(title, error):

    return f"## ❌ {title} Error\n\n`{str(error)}`"


# ============================================================
# MODULE 1: QUALITATIVE ANALYSIS
# ============================================================

def analyze_qualitative(research_data, project_index):

    if not research_data.strip():
        return "⚠️ Please paste qualitative research data first."

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    prompt = f"""
Analyze the supplied qualitative data for this digital health evaluation.

PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
OBJECTIVE: {project['research_objective']}
QUESTIONS: {project['research_questions']}

DATA:
{compact(research_data, 14000)}

Use only supplied evidence. Do not invent quotations.

Produce concise Markdown with:

1. Initial codes
2. Major themes
3. Supporting excerpts (verbatim only)
4. User needs
5. Barriers/pain points
6. Facilitators
7. Design opportunities
8. Interpretation
9. Evidence strength and limitations

Clearly distinguish evidence from researcher interpretation.
"""

    try:

        result = call_ai(prompt, 4000)

        update_project(
            index,
            qualitative_data=research_data,
            qualitative_result=result,
            qualitative=True
        )

        return result

    except Exception as error:

        return ai_error("Qualitative Analysis", error)


# ============================================================
# MODULE 2: QUANTITATIVE ANALYSIS
# ============================================================

def analyze_quantitative(uploaded_file, project_index):

    if uploaded_file is None:
        return None, "⚠️ Please upload a CSV file."

    try:

        df = pd.read_csv(uploaded_file)

        rows, cols = df.shape

        output = [
            "# 📊 Quantitative Research Analysis",
            "",
            "## Dataset Overview",
            f"- **Rows:** {rows}",
            f"- **Variables:** {cols}",
            "",
            "## Variables",
            ", ".join(map(str, df.columns)),
            "",
            "## Descriptive Statistics"
        ]

        numeric = df.select_dtypes(include="number")

        if not numeric.empty:
            output.append(
                numeric.describe().round(2).to_markdown()
            )
        else:
            output.append("No numeric variables detected.")

        output.extend(["", "## Categorical Variables"])

        categorical = df.select_dtypes(exclude="number")

        if categorical.empty:
            output.append("No categorical variables detected.")

        for column in categorical.columns:

            output.append(f"### {column}")

            for value, count in (
                df[column]
                .fillna("Missing")
                .value_counts()
                .head(20)
                .items()
            ):
                output.append(f"- **{value}:** {count}")

        missing = df.isna().sum()

        output.extend(["", "## Missing Data"])

        if missing.sum() == 0:
            output.append("No missing values were detected.")
        else:
            output.extend(
                f"- **{column}:** {count}"
                for column, count in missing[missing > 0].items()
            )

        output.extend([
            "",
            "## Interpretation Note",
            (
                "This descriptive analysis does not establish statistical "
                "significance or causality. Interpret patterns alongside "
                "the qualitative and usability evidence."
            )
        ])

        result = "\n".join(output)

        # Save results
        project, index = get_project(project_index)

        if project:
            update_project(
                index,
                quantitative_result=result,
                quantitative=True
            )

        # Generate charts for Streamlit
        charts = []

        numeric_columns = list(numeric.columns)

        # Histogram
        if numeric_columns:

            col = numeric_columns[0]

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.hist(
                numeric[col].dropna(),
                bins=15,
                color="#4a90d9",
                edgecolor="white",
                alpha=0.85
            )

            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.3)

            charts.append(fig)

        # Correlation heatmap
        if len(numeric_columns) >= 2:

            corr = numeric.corr()

            fig, ax = plt.subplots(
                figsize=(
                    max(6, len(corr.columns) * 0.9),
                    max(5, len(corr.columns) * 0.8)
                )
            )

            im = ax.imshow(
                corr,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                aspect="auto"
            )

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(
                corr.columns,
                rotation=45,
                ha="right"
            )

            ax.set_yticklabels(corr.columns)

            ax.set_title("Correlation Matrix")

            fig.colorbar(im, ax=ax, shrink=0.8)

            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center"
                    )

            charts.append(fig)

        # Missing data chart
        missing_values = df.isna().sum()
        missing_values = missing_values[missing_values > 0]

        if not missing_values.empty:

            fig, ax = plt.subplots(
                figsize=(8, max(4, len(missing_values) * 0.5))
            )

            missing_values.sort_values().plot(
                kind="barh",
                ax=ax,
                color="#e74c3c"
            )

            ax.set_title("Missing Data by Variable")
            ax.set_xlabel("Number of Missing Values")

            charts.append(fig)

        return charts, result

    except Exception as error:

        return None, ai_error("Quantitative Analysis", error)


# ============================================================
# MODULE 3: USER JOURNEY MAPPING
# ============================================================

def generate_journey_map(project_index, qualitative_findings):

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    qualitative_findings = (
        qualitative_findings
        or project.get("qualitative_result", "")
    )

    if not qualitative_findings.strip():
        return "⚠️ Run qualitative analysis or paste qualitative findings first."

    prompt = f"""
Create an evidence-informed user journey map.

PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
TARGET USERS: {project['target_users']}

QUALITATIVE FINDINGS:
{compact(qualitative_findings, 8500)}

Use only supplied evidence. Do not invent experiences or quotes.

Create a Markdown table for:

Awareness, Consideration, Onboarding/First Use,
Regular Use, Problem/Barrier, and Continued Use/Drop-off.

Use columns:

Stage | User action | Experience | Pain point | Need |
Design opportunity | Evidence

Then provide key journey insights, critical pain points,
and research limitations.
"""

    try:

        result = call_ai(prompt, 3000)

        update_project(
            index,
            journey_result=result,
            journey=True
        )

        return result

    except Exception as error:

        return ai_error("Journey Mapping", error)


# ============================================================
# MODULE 4: USABILITY / HEURISTIC EVALUATION
# ============================================================

def analyze_usability(project_index, qualitative_findings, journey_map):

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    qualitative_findings = (
        qualitative_findings
        or project.get("qualitative_result", "")
    )

    journey_map = (
        journey_map
        or project.get("journey_result", "")
    )

    prompt = f"""
Conduct an evidence-informed usability and heuristic evaluation.

INNOVATION: {project['innovation_name']}
DESCRIPTION: {project['innovation_description']}
TARGET USERS: {project['target_users']}

QUALITATIVE EVIDENCE:
{compact(qualitative_findings, 5500)}

JOURNEY EVIDENCE:
{compact(journey_map, 4500)}

Use only supplied material. Never invent participant quotations.

Provide:

1. Table: Problem | Evidence | Severity (Critical/High/Moderate/Low) | Likely impact
2. Navigation/findability
3. Information quality and trust
4. Interaction/feedback
5. Accessibility and inclusion, labeling untested areas
6. Positive features
7. Five priority improvements
8. Evidence limits
"""

    try:

        result = call_ai(prompt, 3200)

        update_project(
            index,
            usability_result=result,
            usability=True
        )

        return result

    except Exception as error:

        return ai_error("Usability Evaluation", error)


# ============================================================
# MODULE 5: MIXED-METHODS SYNTHESIS
# ============================================================

def mixed_methods_synthesis(
    project_index,
    qualitative,
    quantitative,
    usability,
    journey
):

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    qualitative = qualitative or project.get("qualitative_result", "")
    quantitative = quantitative or project.get("quantitative_result", "")
    usability = usability or project.get("usability_result", "")
    journey = journey or project.get("journey_result", "")

    if not qualitative.strip() or not quantitative.strip():
        return "⚠️ Qualitative and quantitative findings are required."

    prompt = f"""
Produce a concise mixed-methods synthesis.

OBJECTIVE: {project['research_objective']}
QUESTIONS: {project['research_questions']}

QUALITATIVE:
{compact(qualitative, 4500)}

QUANTITATIVE:
{compact(quantitative, 3500)}

USABILITY:
{compact(usability, 3000)}

JOURNEY:
{compact(journey, 3000)}

Use only supplied evidence. Do not invent data, statistical significance,
or quotations.

Include:

1. Convergent findings
2. Divergent findings
3. Complementary findings
4. A joint display table
   (issue | qualitative | quantitative | usability/journey | interpretation)
5. Five priority findings
6. Design implications
7. Overall interpretation
8. Limitations and evidence strength
"""

    try:

        result = call_ai(prompt, 2800)

        update_project(
            index,
            mixed_methods_result=result,
            mixed_methods=True
        )

        return result

    except Exception as error:

        return ai_error("Mixed-Methods Synthesis", error)


# ============================================================
# MODULE 6: CO-DESIGN
# ============================================================

def generate_co_design(project_index, participant_brief, mixed_methods):

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    mixed_methods = (
        mixed_methods
        or project.get("mixed_methods_result", "")
    )

    if not mixed_methods.strip():
        return "⚠️ Complete mixed-methods synthesis or paste its findings first."

    prompt = f"""
Design an ethical, practical co-design package for a digital health evaluation.

PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
TARGET USERS: {project['target_users']}

OPTIONAL PARTICIPANT/CONTEXT BRIEF:
{compact(participant_brief, 2500)}

MIXED-METHODS FINDINGS:
{compact(mixed_methods, 8000)}

Use the supplied findings as the basis.
Do not invent stakeholder opinions.

Provide:

1. Purpose
2. Participant groups and inclusion considerations
3. 90-minute session agenda
4. Plain-language activities
5. 5–8 evidence-linked design prompts
6. Prioritisation method
7. Consent, safeguarding, privacy, and accessibility considerations
8. Capture template
9. Expected co-design outputs

Clearly label proposed workshop activities as proposed,
not completed evidence.
"""

    try:

        result = call_ai(prompt, 3200)

        update_project(
            index,
            co_design_brief=participant_brief,
            co_design_result=result,
            co_design=True
        )

        return result

    except Exception as error:

        return ai_error("Co-Design", error)


# ============================================================
# MODULE 7: EVIDENCE VERIFICATION / AUDIT
# ============================================================

def evidence_audit(
    project_index,
    mixed_methods,
    co_design,
    external_evidence
):

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    mixed_methods = (
        mixed_methods
        or project.get("mixed_methods_result", "")
    )

    co_design = (
        co_design
        or project.get("co_design_result", "")
    )

    if not mixed_methods.strip():
        return "⚠️ Mixed-methods findings are required for an evidence audit."

    prompt = f"""
Perform an evidence traceability and quality audit
for this digital health evaluation.

PROJECT: {project['project_name']}
RESEARCH QUESTIONS: {project['research_questions']}

MIXED-METHODS FINDINGS:
{compact(mixed_methods, 7000)}

CO-DESIGN PLAN/OUTPUT:
{compact(co_design, 4000)}

EXTERNAL EVIDENCE PROVIDED BY RESEARCHER:
{compact(external_evidence, 4000)}

Audit only the supplied material.

Do not claim external verification, literature review,
or guideline compliance unless explicitly supplied.

Produce:

1. Traceability table:
   Claim/Proposed action | Supporting source | Evidence type |
   Strength (High/Moderate/Limited) | Audit judgement

2. Unsupported/overstated claims to revise
3. Gaps requiring further evidence or testing
4. Data-quality and equity considerations
5. Audit-ready wording for conclusions
6. Next verification steps
"""

    try:

        result = call_ai(prompt, 3200)

        update_project(
            index,
            evidence_audit_result=result,
            evidence_audit=True
        )

        return result

    except Exception as error:

        return ai_error("Evidence Verification / Audit", error)


# ============================================================
# MODULE 8: EVIDENCE-BASED RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    project_index,
    mixed_methods,
    co_design,
    evidence_audit_result
):

    project, index = get_project(project_index)

    if not project:
        return "⚠️ Select a project first."

    mixed_methods = (
        mixed_methods
        or project.get("mixed_methods_result", "")
    )

    co_design = (
        co_design
        or project.get("co_design_result", "")
    )

    evidence_audit_result = (
        evidence_audit_result
        or project.get("evidence_audit_result", "")
    )

    if not mixed_methods.strip() or not evidence_audit_result.strip():
        return (
            "⚠️ Complete mixed-methods synthesis and "
            "evidence verification/audit first."
        )

    prompt = f"""
Create final evidence-based recommendations
for a digital health innovation evaluation.

PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
OBJECTIVE: {project['research_objective']}

MIXED-METHODS SYNTHESIS:
{compact(mixed_methods, 6500)}

CO-DESIGN MATERIAL:
{compact(co_design, 3500)}

EVIDENCE AUDIT:
{compact(evidence_audit_result, 6000)}

Use only supplied evidence and preserve the audit caveats.

Provide:

1. Executive summary
2. Prioritized recommendation table with:
   Recommendation | Evidence rationale | Priority | Effort |
   Owner | Success measure
3. Immediate (0–3 months), near-term (3–6 months),
   and longer-term (6–12 months) roadmap
4. Safety, equity, privacy and trust considerations
5. Evaluation metrics
6. Implementation risks/mitigations
7. Conclusion

Do not state that proposed co-design activities have occurred
unless actual outcomes are supplied.
"""

    try:

        result = call_ai(prompt, 3600)

        update_project(
            index,
            recommendations_result=result,
            recommendations=True
        )

        return result

    except Exception as error:

        return ai_error("Evidence-Based Recommendations", error)
        # ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.sidebar.title("🧠 Digital Health AI")
st.sidebar.caption("Generative AI-Assisted Research Evaluation")

projects = load_projects()

if not projects:
    st.sidebar.warning("No research projects available.")

# Project selector
if projects:

    selected_project_index = st.sidebar.selectbox(
        "Select Research Project",
        options=list(range(len(projects))),
        format_func=lambda i: (
            projects[i].get("project_name")
            or projects[i].get("innovation_name")
            or f"Project {i + 1}"
        ),
        key="project_selector"
    )

    selected_project, selected_project_index = get_project(
        selected_project_index
    )

else:
    selected_project_index = None
    selected_project = None


# Navigation
page_options = [
    "🏠 Home",
    "📊 Dashboard",
    "➕ Create Project",
    "🧠 Project Workspace",
    "📝 Qualitative Analysis",
    "📊 Quantitative Analysis",
    "🔍 Usability Evaluation",
    "🧭 User Journey Mapping",
    "🔀 Mixed-Methods Synthesis",
    "🤝 Co-Design",
    "✅ Evidence Audit",
    "💡 Recommendations",
]

page = st.sidebar.radio(
    "Navigation",
    page_options,
    key="main_navigation"
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-assisted research support. "
    "All findings require researcher review and verification."
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("Digital Health AI Research Platform")

    st.subheader(
        "Generative AI-Assisted Mixed-Methods Evaluation"
    )

    st.markdown("""
    A researcher-focused workspace for transforming qualitative,
    quantitative, usability and co-design evidence into structured,
    traceable research insights and evidence-based recommendations.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Research Data")
        st.write(
            "Organize qualitative and quantitative research evidence."
        )

        st.markdown("### 🔍 Evaluation")
        st.write(
            "Explore usability, user experience and user journeys."
        )

    with col2:
        st.markdown("### 🤝 Co-Design")
        st.write(
            "Develop evidence-linked co-design activities."
        )

        st.markdown("### 💡 Research Insights")
        st.write(
            "Synthesize evidence and develop practical recommendations."
        )

    st.divider()

    st.info(
        "To enter your research workspace, select Dashboard "
        "from the Navigation menu."
    )
