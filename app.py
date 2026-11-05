import os
import json
import tempfile
from datetime import datetime

import pandas as pd
import gradio as gr
import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from groq import Groq


MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PROJECTS_FILE = "research_projects.json"

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
# STORAGE AND PROJECT STATE
# ============================================================

def load_projects():
    if not os.path.exists(PROJECTS_FILE):
        return []
    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_projects(projects):
    with open(PROJECTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(projects, handle, indent=2, ensure_ascii=False)


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
    """Create a realistic starting project only for a new installation."""
    if os.path.exists(PROJECTS_FILE):
        return
    demo = blank_project(
        project_name="ChildHealth Connect Evaluation",
        innovation_name="ChildHealth Connect",
        innovation_description=(
            "A digital health platform designed to help parents and caregivers access "
            "understandable child-health information, reminders, progress tracking, "
            "and opportunities to provide feedback."
        ),
        target_users="Parents and caregivers of young children",
        research_objective=(
            "To evaluate usability, user experience, acceptability, trust, information "
            "needs, and barriers to engagement with ChildHealth Connect."
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
    """Safely migrate projects created with earlier versions of the app."""
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


def project_choices():
    return [
        (f"{number + 1}. {project.get('project_name') or project.get('innovation_name') or 'Untitled project'}", number)
        for number, project in enumerate(load_projects())
    ]


def module_status(project, key):
    return "🟢 Complete" if project.get(key) else "🟡 Not completed"


def project_card(project, index):
    project, _ = normalize_project(project)
    module_lines = "\n".join(
        f"- {label}: **{'✓' if project.get(key) else '○'}**" for key, label in MODULES
    )
    return f"""### {index + 1}. {project.get('innovation_name') or project.get('project_name')}
**{project.get('evaluation_type', 'Digital Health Research')}**  
👥 {project.get('target_users') or 'Target users not specified'}

{module_lines}
"""


def dashboard_projects():
    projects = load_projects()
    if not projects:
        return "### 📁 My Research Projects\n\nNo projects yet. Create your first research project.", "### 📁 Projects\n# 0", gr.update(choices=[], value=None)
    cards = ["### 📁 My Research Projects"]
    cards.extend(project_card(project, i) for i, project in enumerate(projects))
    return "\n---\n".join(cards), f"### 📁 Projects\n# {len(projects)}", gr.update(choices=project_choices(), value=0)


def workspace_markdown(project):
    if not project:
        return "## No project selected."
    rows = "\n".join(f"| {label} | {module_status(project, key)} |" for key, label in MODULES)
    return f"""# 🧠 {project['project_name']}

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
**Project Setup → Qualitative → Quantitative → Usability → Journey Mapping → Mixed-Methods → Co-Design → Evidence Verification/Audit → Evidence-Based Recommendations**

| Module | Status |
|---|---|
{rows}
"""


def create_project(project_name, innovation_name, innovation_description, target_users,
                   research_objective, research_questions, evaluation_type):
    if not (project_name or "").strip():
        return "⚠️ Please enter a project name.", gr.update(), gr.update(), gr.update()
    if not (innovation_name or "").strip():
        return "⚠️ Please enter the digital health innovation name.", gr.update(), gr.update(), gr.update()
    projects = load_projects()
    project = blank_project(
        project_name=project_name.strip(), innovation_name=innovation_name.strip(),
        innovation_description=(innovation_description or "").strip(),
        target_users=(target_users or "").strip(),
        research_objective=(research_objective or "").strip(),
        research_questions=(research_questions or "").strip(), evaluation_type=evaluation_type,
    )
    projects.append(project)
    save_projects(projects)
    cards, count, selector = dashboard_projects()
    return f"### ✅ Project created: **{project['project_name']}**", cards, count, selector


# ============================================================
# AI HELPERS
# ============================================================

def get_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it in Hugging Face Settings → Secrets.")
    return Groq(api_key=GROQ_API_KEY)


def call_ai(prompt, max_tokens=3000):
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You are a careful digital health research assistant. Never fabricate evidence, quotations, participants, statistics, citations, or causal claims."}, {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def ai_error(title, error):
    return f"## ❌ {title} Error\n\n`{str(error)}`"


def compact(text, limit):
    return (text or "").strip()[:limit]


# ============================================================
# ANALYSIS MODULES
# ============================================================

def analyze_qualitative(research_data, project_index):
    if not (research_data or "").strip():
        return "⚠️ Please paste qualitative research data first.", "", gr.update()
    project, index = get_project(project_index)
    if not project:
        return "⚠️ Select a project first.", "", gr.update()
    prompt = f"""Analyze the supplied qualitative data for this digital health evaluation.

PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
OBJECTIVE: {project['research_objective']}
QUESTIONS: {project['research_questions']}
DATA:\n{compact(research_data, 14000)}

Use only supplied evidence. Do not invent quotations. Produce concise Markdown with:
1. Initial codes; 2. Major themes; 3. Supporting excerpts (verbatim only); 4. User needs;
5. Barriers/pain points; 6. Facilitators; 7. Design opportunities; 8. Interpretation;
9. Evidence strength and limitations. Clearly distinguish evidence from researcher interpretation."""
    try:
        result = call_ai(prompt, 4000)
        update_project(index, qualitative_data=research_data, qualitative_result=result, qualitative=True)
        return result, result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Qualitative Analysis", error), "", gr.update()


def analyze_quantitative(file_path, project_index):
    if not file_path:
        return "⚠️ Please upload a CSV file.", "", [], gr.update()
    try:
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        output = ["# 📊 Quantitative Research Analysis", "", "## Dataset Overview", f"- **Rows:** {rows}", f"- **Variables:** {cols}", "", "## Variables", ", ".join(map(str, df.columns)), "", "## Descriptive Statistics"]
        numeric = df.select_dtypes(include="number")
        output.append(numeric.describe().round(2).to_html() if not numeric.empty else "No numeric variables detected.")
        output.extend(["", "## Categorical Variables"])
        categorical = df.select_dtypes(exclude="number")
        if categorical.empty:
            output.append("No categorical variables detected.")
        for column in categorical.columns:
            output.append(f"### {column}")
            for value, count in df[column].fillna("Missing").value_counts().head(20).items():
                output.append(f"- **{value}:** {count}")
        missing = df.isna().sum()
        output.extend(["", "## Missing Data"])
        output.append("No missing values were detected." if missing.sum() == 0 else "\n".join(f"- **{column}:** {count}" for column, count in missing[missing > 0].items()))
        output.extend(["", "## Interpretation Note", "This descriptive analysis does not establish statistical significance or causality. Interpret patterns alongside the qualitative and usability evidence."])
        result = "\n".join(output)

        # --------------------------------------------------------
        # CHART GENERATION
        # --------------------------------------------------------
        chart_files = []
        numeric_columns = list(numeric.columns)
        categorical_columns = list(categorical.columns)

        # 1. Histogram for first numeric variable
        if numeric_columns:
            col = numeric_columns[0]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(numeric[col].dropna(), bins=15, color="#4a90d9", edgecolor="white", alpha=0.85)
            ax.set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            path = os.path.join(tempfile.gettempdir(), "quant_hist.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            chart_files.append(path)

        # 2. Correlation heatmap

        if len(numeric_columns) >= 2:
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)*0.9), max(5, len(corr.columns)*0.8)))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(corr.columns, fontsize=10)
            ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Correlation", rotation=270, labelpad=18)
            # Add text annotations
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                                   color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=9)
            path = os.path.join(tempfile.gettempdir(), "quant_corr.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            chart_files.append(path)

        # 3. Missing data overview
        missing_vals = df.isna().sum()
        missing_vals = missing_vals[missing_vals > 0]
        if not missing_vals.empty:
            fig, ax = plt.subplots(figsize=(8, max(4, len(missing_vals)*0.5)))
            missing_vals.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#e74c3c", edgecolor="white")
            ax.set_title("Missing Data by Variable", fontsize=14, fontweight="bold")
            ax.set_xlabel("Number of Missing Values", fontsize=12)
            ax.grid(axis="x", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            path = os.path.join(tempfile.gettempdir(), "quant_missing.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            chart_files.append(path)

        project, index = get_project(project_index)
        if project:
            update_project(index, quantitative_result=result, quantitative=True)
            return result, result, chart_files, gr.update(value=workspace_markdown(get_project(index)[0]))
        return result, result, chart_files, gr.update()
    except Exception as error:
        return ai_error("Quantitative Analysis", error), "", [], gr.update()


def generate_journey_map(project_index, qualitative_findings):
    project, index = get_project(project_index)
    qualitative_findings = qualitative_findings or (project or {}).get("qualitative_result", "")
    if not project:
        return "⚠️ Select a project first.", gr.update()
    if not qualitative_findings.strip():
        return "⚠️ Run qualitative analysis or paste qualitative findings first.", gr.update()
    prompt = f"""Create an evidence-informed user journey map.
PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
TARGET USERS: {project['target_users']}
QUALITATIVE FINDINGS:\n{compact(qualitative_findings, 8500)}

Use only supplied evidence. Do not invent experiences or quotes. Create a Markdown table for Awareness, Consideration, Onboarding/First Use, Regular Use, Problem/Barrier, and Continued Use/Drop-off with columns: Stage | User action | Experience | Pain point | Need | Design opportunity | Evidence. Then provide key journey insights, critical pain points, and research limitations."""
    try:
        result = call_ai(prompt, 3000)
        update_project(index, journey_result=result, journey=True)
        return result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Journey Mapping", error), gr.update()


def analyze_usability(project_index, qualitative_findings, journey_map):
    project, index = get_project(project_index)
    if not project:
        return "⚠️ Select a project first.", gr.update()
    qualitative_findings = qualitative_findings or project.get("qualitative_result", "")
    journey_map = journey_map or project.get("journey_result", "")
    prompt = f"""Conduct an evidence-informed usability and heuristic evaluation.
INNOVATION: {project['innovation_name']}
DESCRIPTION: {project['innovation_description']}
TARGET USERS: {project['target_users']}
QUALITATIVE EVIDENCE:\n{compact(qualitative_findings, 5500)}
JOURNEY EVIDENCE:\n{compact(journey_map, 4500)}

Use only supplied material. Never invent participant quotations. Provide: (1) table Problem | Evidence | Severity (Critical/High/Moderate/Low) | Likely impact; (2) navigation/findability; (3) information quality and trust; (4) interaction/feedback; (5) accessibility and inclusion, labeling untested areas; (6) positive features; (7) five priority improvements; (8) evidence limits."""
    try:
        result = call_ai(prompt, 3200)
        update_project(index, usability_result=result, usability=True)
        return result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Usability Evaluation", error), gr.update()


def mixed_methods_synthesis(project_index, qualitative, quantitative, usability, journey):
    project, index = get_project(project_index)
    if not project:
        return "⚠️ Select a project first.", gr.update()
    qualitative = qualitative or project.get("qualitative_result", "")
    quantitative = quantitative or project.get("quantitative_result", "")
    usability = usability or project.get("usability_result", "")
    journey = journey or project.get("journey_result", "")
    if not qualitative.strip() or not quantitative.strip():
        return "⚠️ Qualitative and quantitative findings are required.", gr.update()
    prompt = f"""Produce a concise mixed-methods synthesis.
OBJECTIVE: {project['research_objective']}
QUESTIONS: {project['research_questions']}
QUALITATIVE:\n{compact(qualitative, 4500)}
QUANTITATIVE:\n{compact(quantitative, 3500)}
USABILITY:\n{compact(usability, 3000)}
JOURNEY:\n{compact(journey, 3000)}

Use only supplied evidence. Do not invent data, statistical significance, or quotations. Include: convergent findings; divergent findings; complementary findings; a joint display table (issue | qualitative | quantitative | usability/journey | interpretation); five priority findings; design implications; overall interpretation; limitations and evidence strength."""
    try:
        result = call_ai(prompt, 2800)
        update_project(index, mixed_methods_result=result, mixed_methods=True)
        return result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Mixed-Methods Synthesis", error), gr.update()


def generate_co_design(project_index, participant_brief, mixed_methods):
    project, index = get_project(project_index)
    if not project:
        return "⚠️ Select a project first.", gr.update()
    mixed_methods = mixed_methods or project.get("mixed_methods_result", "")
    if not mixed_methods.strip():
        return "⚠️ Complete mixed-methods synthesis or paste its findings first.", gr.update()
    prompt = f"""Design an ethical, practical co-design package for a digital health evaluation.
PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
TARGET USERS: {project['target_users']}
OPTIONAL PARTICIPANT/CONTEXT BRIEF:\n{compact(participant_brief, 2500)}
MIXED-METHODS FINDINGS:\n{compact(mixed_methods, 8000)}

Use the supplied findings as the basis. Do not invent stakeholder opinions. Provide: purpose; participant groups and inclusion considerations; 90-minute session agenda; plain-language activities; 5–8 evidence-linked design prompts; prioritisation method; consent, safeguarding, privacy, and accessibility considerations; capture template; and expected co-design outputs. Clearly label proposed workshop activities as proposed, not completed evidence."""
    try:
        result = call_ai(prompt, 3200)
        update_project(index, co_design_brief=participant_brief, co_design_result=result, co_design=True)
        return result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Co-Design", error), gr.update()


def evidence_audit(project_index, mixed_methods, co_design, external_evidence):
    project, index = get_project(project_index)
    if not project:
        return "⚠️ Select a project first.", gr.update()
    mixed_methods = mixed_methods or project.get("mixed_methods_result", "")
    co_design = co_design or project.get("co_design_result", "")
    if not mixed_methods.strip():
        return "⚠️ Mixed-methods findings are required for an evidence audit.", gr.update()
    prompt = f"""Perform an evidence traceability and quality audit for this digital health evaluation.
PROJECT: {project['project_name']}
RESEARCH QUESTIONS: {project['research_questions']}
MIXED-METHODS FINDINGS:\n{compact(mixed_methods, 7000)}
CO-DESIGN PLAN/OUTPUT:\n{compact(co_design, 4000)}
EXTERNAL EVIDENCE PROVIDED BY RESEARCHER:\n{compact(external_evidence, 4000)}

Audit only the supplied material. Do not claim external verification, literature review, or guideline compliance unless explicitly supplied. Produce: (1) traceability table Claim/Proposed action | Supporting source | Evidence type | Strength (High/Moderate/Limited) | Audit judgement; (2) unsupported/overstated claims to revise; (3) gaps requiring further evidence or testing; (4) data-quality and equity considerations; (5) audit-ready wording for conclusions; (6) next verification steps."""
    try:
        result = call_ai(prompt, 3200)
        update_project(index, evidence_audit_result=result, evidence_audit=True)
        return result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Evidence Verification / Audit", error), gr.update()


def generate_recommendations(project_index, mixed_methods, co_design, evidence_audit):
    project, index = get_project(project_index)
    if not project:
        return "⚠️ Select a project first.", gr.update()
    mixed_methods = mixed_methods or project.get("mixed_methods_result", "")
    co_design = co_design or project.get("co_design_result", "")
    evidence_audit = evidence_audit or project.get("evidence_audit_result", "")
    if not mixed_methods.strip() or not evidence_audit.strip():
        return "⚠️ Complete mixed-methods synthesis and evidence verification/audit first.", gr.update()
    prompt = f"""Create final evidence-based recommendations for a digital health innovation evaluation.
PROJECT: {project['project_name']}
INNOVATION: {project['innovation_name']}
OBJECTIVE: {project['research_objective']}
MIXED-METHODS SYNTHESIS:\n{compact(mixed_methods, 6500)}
CO-DESIGN MATERIAL:\n{compact(co_design, 3500)}
EVIDENCE AUDIT:\n{compact(evidence_audit, 6000)}

Use only supplied evidence and preserve the audit caveats. Provide: executive summary; prioritized recommendation table with Recommendation | Evidence rationale | Priority | Effort | Owner | Success measure; immediate (0–3 months), near-term (3–6 months), and longer-term (6–12 months) roadmap; safety, equity, privacy and trust considerations; evaluation metrics; implementation risks/mitigations; and a conclusion. Do not state that proposed co-design activities have occurred unless actual outcomes are supplied."""
    try:
        result = call_ai(prompt, 3600)
        update_project(index, recommendations_result=result, recommendations=True)
        return result, gr.update(value=workspace_markdown(get_project(index)[0]))
    except Exception as error:
        return ai_error("Evidence-Based Recommendations", error), gr.update()


# ============================================================
# UI HELPERS
# ============================================================

def open_project(index):
    project, index = get_project(index)
    if not project:
        return gr.update(visible=True), gr.update(visible=False), "## No project found."
    return gr.update(visible=False), gr.update(visible=True), workspace_markdown(project)


def populate_module(project_index):
    """Load all persisted project inputs and outputs into module controls."""
    project, _ = get_project(project_index)
    project = project or blank_project()
    qualitative = project.get("qualitative_result", "")
    quantitative = project.get("quantitative_result", "")
    usability = project.get("usability_result", "")
    journey = project.get("journey_result", "")
    mixed = project.get("mixed_methods_result", "")
    co_design = project.get("co_design_result", "")
    audit = project.get("evidence_audit_result", "")
    recommendations = project.get("recommendations_result", "")
    return (
        project.get("qualitative_data", ""), qualitative,
        journey, journey,
        qualitative, journey, usability,
        quantitative,
        qualitative, quantitative, usability, journey, mixed,
        project.get("co_design_brief", ""), mixed, co_design,
        mixed, co_design, audit,
        mixed, co_design, audit, recommendations,
        workspace_markdown(project),
    )


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(title="Digital Health AI Research Platform") as demo:
    selected_project = gr.State(0)

    # ========================================================
    # LANDING PAGE (from your uploaded code)
    # ========================================================

    with gr.Column(visible=True, elem_id="landing_page") as landing_screen:
        gr.HTML("""
        <style>
        #landing_page {
            min-height: 90vh;
            padding: 35px 7%;
        }
        .hero {
            max-width: 1250px;
            margin: auto;
            padding: 25px 0;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 0.9fr 1.1fr;
            gap: 45px;
            align-items: center;
        }
        .eyebrow {
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 2px;
            text-transform: uppercase;
            opacity: 0.58;
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: clamp(42px, 5vw, 66px);
            line-height: 1.04;
            font-weight: 800;
            margin-bottom: 20px;
        }
        .hero-subtitle {
            font-size: 22px;
            line-height: 1.4;
            opacity: 0.72;
            margin-bottom: 20px;
        }
        .hero-description {
            font-size: 16px;
            line-height: 1.65;
            opacity: 0.65;
            max-width: 550px;
        }
        .visual-card {
            border-radius: 30px;
            padding: 18px;
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(120,120,120,0.15);
            box-shadow: 0 20px 60px rgba(0,0,0,0.08);
        }
        .research-illustration {
            width: 100%;
            height: auto;
            display: block;
        }
        .visual-caption {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 12px;
            font-size: 12px;
            font-weight: 600;
            opacity: 0.6;
            flex-wrap: wrap;
        }
        .bottom-line {
            text-align: center;
            margin-top: 40px;
            font-size: 19px;
            font-weight: 650;
            opacity: 0.6;
        }
        .bottom-line span {
            font-weight: 800;
            opacity: 1;
        }
        @media (max-width: 850px) {
            .hero-grid { grid-template-columns: 1fr; }
            .hero-title { font-size: 44px; }
        }
        </style>
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">DIGITAL HEALTH RESEARCH</div>
                    <div class="hero-title">Digital Health<br>AI Research Platform</div>
                    <div class="hero-subtitle">Generative AI–Assisted<br>Mixed-Methods Evaluation</div>
                    <div class="hero-description">
                        A researcher-focused workspace for transforming
                        qualitative, quantitative, usability and co-design
                        evidence into structured, traceable research insights
                        and evidence-based recommendations.
                    </div>
                </div>
                <div class="visual-card">
                    <svg viewBox="0 0 620 520" xmlns="http://www.w3.org/2000/svg" class="research-illustration">
                        <rect x="20" y="20" width="580" height="480" rx="35" fill="#f7f9fc"/>
                        <circle cx="520" cy="85" r="55" fill="#e8f0ff"/>
                        <circle cx="105" cy="425" r="65" fill="#eef9f3"/>
                        <circle cx="310" cy="150" r="70" fill="#ffffff" stroke="#d9e2f2" stroke-width="3"/>
                        <text x="310" y="145" text-anchor="middle" font-size="48">🧠</text>
                        <text x="310" y="180" text-anchor="middle" font-size="15" font-weight="600" fill="#536174">AI RESEARCH ASSISTANT</text>
                        <circle cx="125" cy="205" r="30" fill="#dce9ff"/>
                        <text x="125" y="218" text-anchor="middle" font-size="38">👩‍🔬</text>
                        <text x="125" y="255" text-anchor="middle" font-size="14" font-weight="600" fill="#536174">RESEARCHER</text>
                        <rect x="55" y="305" width="145" height="75" rx="16" fill="#ffffff" stroke="#e0e6ef"/>
                        <text x="78" y="337" font-size="25">📝</text>
                        <text x="112" y="335" font-size="13" font-weight="700" fill="#39475a">QUALITATIVE</text>
                        <text x="112" y="355" font-size="11" fill="#778397">Themes &amp; Insights</text>
                        <rect x="420" y="305" width="145" height="75" rx="16" fill="#ffffff" stroke="#e0e6ef"/>
                        <text x="443" y="338" font-size="25">📊</text>
                        <text x="478" y="335" font-size="13" font-weight="700" fill="#39475a">QUANTITATIVE</text>
                        <text x="478" y="355" font-size="11" fill="#778397">Patterns &amp; Measures</text>
                        <path d="M155 235 C190 250 220 245 250 205" fill="none" stroke="#9eb8df" stroke-width="3"/>
                        <path d="M370 205 C405 245 430 250 455 305" fill="none" stroke="#9eb8df" stroke-width="3"/>
                        <path d="M250 150 C215 125 185 125 160 155" fill="none" stroke="#b8c9df" stroke-width="3"/>
                        <rect x="225" y="265" width="170" height="70" rx="20" fill="#ffffff" stroke="#dce4ef" stroke-width="2"/>
                        <text x="310" y="296" text-anchor="middle" font-size="28">💡</text>
                        <text x="310" y="321" text-anchor="middle" font-size="13" font-weight="700" fill="#39475a">EVIDENCE → INSIGHT</text>
                        <rect x="205" y="400" width="210" height="58" rx="18" fill="#ffffff" stroke="#dce4ef"/>
                        <text x="310" y="425" text-anchor="middle" font-size="25">✅</text>
                        <text x="310" y="446" text-anchor="middle" font-size="12" font-weight="700" fill="#39475a">EVIDENCE-BASED ACTION</text>
                        <path d="M310 335 L310 400" stroke="#9eb8df" stroke-width="3"/>
                    </svg>
                    <div class="visual-caption">
                        <span>Research Data</span><span>→</span><span>AI-Assisted Analysis</span><span>→</span><span>Evidence-Based Action</span>
                    </div>
                </div>
            </div>
            <div class="bottom-line">
                From <span>Evidence</span> → <span>Insight</span> → <span>Action</span>
            </div>
        </div>
        """)
        enter_workspace = gr.Button("ENTER RESEARCH WORKSPACE  →", variant="primary", size="lg")

    # ========================================================
    # DASHBOARD (original app.py)
    # ========================================================

    with gr.Column(visible=False) as dashboard:
        gr.Markdown("# 🧠 Digital Health AI Research Platform\n## Researcher Dashboard")
        with gr.Row():
            project_count = gr.Markdown("### 📁 Projects\n# 0")
            gr.Markdown("### 📝 Analyses\n# AI-assisted")
            gr.Markdown("### 👥 Participants\n# Study-led")
            gr.Markdown("### 💡 Insights\n# Evidence-led")
        saved_projects = gr.Markdown()
        project_selector = gr.Dropdown(label="Select a saved project", choices=project_choices(), value=0 if load_projects() else None, interactive=True)
        with gr.Row():
            open_selected_project = gr.Button("📂 OPEN SELECTED PROJECT", variant="primary")
            create_project_button = gr.Button("➕ CREATE NEW RESEARCH PROJECT", variant="primary")
        gr.Markdown("**Workflow:** Project Setup → Qualitative → Quantitative → Usability → Journey Mapping → Mixed-Methods → Co-Design → Evidence Verification/Audit → Evidence-Based Recommendations")
        system_status = gr.Textbox(label="System Status", value="Ready", interactive=False)
        with gr.Row():
            system_button = gr.Button("⚙️ Check System")
            logout_button = gr.Button("🚪 Back to Home")

    with gr.Column(visible=False) as project_setup:
        gr.Markdown("# ➕ Create New Research Project\n### Step 1 — Define Your Research Project")
        project_name = gr.Textbox(label="Project Name *")
        innovation_name = gr.Textbox(label="Digital Health Innovation *")
        innovation_description = gr.Textbox(label="Innovation Description", lines=4)
        target_users = gr.Textbox(label="Target Users")
        research_objective = gr.Textbox(label="Research Objective", lines=4)
        research_questions = gr.Textbox(label="Research Questions", lines=6)
        evaluation_type = gr.Dropdown(["Qualitative Evaluation", "Usability Evaluation", "User Experience Evaluation", "Acceptability Evaluation", "Feasibility Evaluation", "Digital Health Intervention Evaluation", "Mixed-Methods Evaluation"], value="Mixed-Methods Evaluation", label="Evaluation Type")
        create_message = gr.Markdown()
        with gr.Row():
            create_button = gr.Button("🚀 CREATE PROJECT", variant="primary")
            cancel_button = gr.Button("← Back to Dashboard")

    with gr.Column(visible=False) as project_workspace:
        project_output = gr.Markdown()
        gr.Markdown("---\n## 🔬 Research Modules")
        with gr.Row():
            qualitative_button = gr.Button("📝 QUALITATIVE", variant="primary")
            quantitative_button = gr.Button("📊 QUANTITATIVE", variant="primary")
            usability_button = gr.Button("🔍 USABILITY", variant="primary")
            journey_button = gr.Button("🧭 JOURNEY MAP", variant="primary")
        with gr.Row():
            mixed_methods_button = gr.Button("🔀 MIXED-METHODS", variant="primary")
            co_design_button = gr.Button("🤝 CO-DESIGN", variant="primary")
            audit_button = gr.Button("✅ EVIDENCE AUDIT", variant="primary")
            recommendations_button = gr.Button("💡 RECOMMENDATIONS", variant="primary")
        back_dashboard_button = gr.Button("← Back to Dashboard")

    with gr.Column(visible=False) as qualitative_page:
        gr.Markdown("# 📝 Qualitative Research Analysis")
        research_data = gr.Textbox(label="Research Data", lines=18, placeholder="Paste interviews, focus groups, observations, or field notes.")
        analyze_button = gr.Button("🤖 ANALYZE WITH AI", variant="primary")
        qualitative_result = gr.Markdown()
        qual_back = gr.Button("← Back to Project Workspace")

    with gr.Column(visible=False) as quantitative_page:
        gr.Markdown("# 📊 Quantitative Research Analysis\nUpload a CSV dataset for descriptive analysis.")
        quantitative_file = gr.File(label="Upload CSV Dataset", file_types=[".csv"], type="filepath")
        quantitative_button_run = gr.Button("📊 ANALYZE QUANTITATIVE DATA", variant="primary")
        quantitative_result = gr.Markdown()
        quantitative_gallery = gr.Gallery(
            label="📈 Generated Research Visualisations",
            columns=2,
            height="auto"
        )
        quant_back = gr.Button("← Back to Project Workspace")


    with gr.Column(visible=False) as usability_page:
        gr.Markdown("# 🔍 Usability / Heuristic Evaluation")
        usability_qualitative = gr.Textbox(label="Qualitative Findings (auto-filled from project)", lines=10)
        usability_journey = gr.Textbox(label="User Journey Map (auto-filled from project)", lines=10)
        usability_run = gr.Button("🤖 ANALYZE USABILITY", variant="primary")
        usability_result = gr.Markdown()
        usability_back = gr.Button("← Back to Project Workspace")

    with gr.Column(visible=False) as journey_page:
        gr.Markdown("# 🧭 User Journey Mapping")
        journey_qualitative = gr.Textbox(label="Qualitative Findings (auto-filled from project)", lines=12)
        journey_run = gr.Button("🧭 GENERATE JOURNEY MAP", variant="primary")
        journey_result = gr.Markdown()
        journey_back = gr.Button("← Back to Project Workspace")

    with gr.Column(visible=False) as mixed_page:
        gr.Markdown("# 🔀 Mixed-Methods Synthesis")
        mm_qualitative = gr.Textbox(label="Qualitative Findings", lines=10)
        mm_quantitative = gr.Textbox(label="Quantitative Findings", lines=10)
        mm_usability = gr.Textbox(label="Usability Findings", lines=8)
        mm_journey = gr.Textbox(label="Journey Map", lines=8)
        mixed_run = gr.Button("🔀 GENERATE MIXED-METHODS SYNTHESIS", variant="primary")
        mixed_result = gr.Markdown()
        mixed_back = gr.Button("← Back to Project Workspace")

    with gr.Column(visible=False) as co_design_page:
        gr.Markdown("# 🤝 Co-Design\nDevelop an evidence-linked workshop and prioritisation plan with parents/caregivers.")
        co_design_brief = gr.Textbox(label="Participant / Context Brief (optional)", lines=6, placeholder="Add recruitment context, accessibility needs, existing participant feedback, or practical constraints.")
        co_design_mixed = gr.Textbox(label="Mixed-Methods Findings (auto-filled from project)", lines=12)
        co_design_run = gr.Button("🤝 GENERATE CO-DESIGN PACKAGE", variant="primary")
        co_design_result = gr.Markdown()
        co_design_back = gr.Button("← Back to Project Workspace")

    with gr.Column(visible=False) as audit_page:
        gr.Markdown("# ✅ Evidence Verification / Audit\nAudit traceability, evidence strength, claims, gaps, and equity considerations before recommendations.")
        audit_mixed = gr.Textbox(label="Mixed-Methods Findings", lines=10)
        audit_co_design = gr.Textbox(label="Co-Design Plan / Outcomes", lines=8)
        audit_external = gr.Textbox(label="External Evidence Supplied by Researcher (optional)", lines=8, placeholder="Paste verified guideline, literature, policy, or implementation evidence. Nothing is verified unless provided here.")
        audit_run = gr.Button("✅ RUN EVIDENCE AUDIT", variant="primary")
        audit_result = gr.Markdown()
        audit_back = gr.Button("← Back to Project Workspace")

    with gr.Column(visible=False) as recommendations_page:
        gr.Markdown("# 💡 Evidence-Based Recommendations\nGenerate an implementation-ready, evidence-traceable roadmap.")
        rec_mixed = gr.Textbox(label="Mixed-Methods Findings", lines=9)
        rec_co_design = gr.Textbox(label="Co-Design Plan / Outcomes", lines=7)
        rec_audit = gr.Textbox(label="Evidence Audit", lines=10)
        recommendations_run = gr.Button("💡 GENERATE FINAL RECOMMENDATIONS", variant="primary")
        recommendations_result = gr.Markdown()
        recommendations_back = gr.Button("← Back to Project Workspace")

    # ========================================================
    # EVENT HANDLERS
    # ========================================================

    @spaces.GPU
    def gpu_check():
        return "🟢 System is running successfully."

    def enter_dashboard():
        cards, count, selector = dashboard_projects()
        return (
            gr.update(visible=False),   # landing_screen
            gr.update(visible=True),    # dashboard
            cards,                      # saved_projects
            count,                      # project_count
            selector,                   # project_selector
        )

    enter_workspace.click(
        enter_dashboard,
        outputs=[landing_screen, dashboard, saved_projects, project_count, project_selector]
    )

    logout_button.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[landing_screen, dashboard]
    )

    system_button.click(gpu_check, outputs=system_status)
    create_project_button.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[dashboard, project_setup])
    cancel_button.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[project_setup, dashboard])
    create_button.click(create_project, [project_name, innovation_name, innovation_description, target_users, research_objective, research_questions, evaluation_type], [create_message, saved_projects, project_count, project_selector])
    open_selected_project.click(open_project, project_selector, [dashboard, project_workspace, project_output])
    back_dashboard_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[dashboard, project_workspace])

    # Project selection keeps data loaded across module pages.
    project_selector.change(lambda value: value, project_selector, selected_project)
    project_selector.change(
        populate_module,
        project_selector,
        [
            research_data, qualitative_result,
            journey_qualitative, journey_result,
            usability_qualitative, usability_journey, usability_result,
            quantitative_result,
            mm_qualitative, mm_quantitative, mm_usability, mm_journey, mixed_result,
            co_design_brief, co_design_mixed, co_design_result,
            audit_mixed, audit_co_design, audit_result,
            rec_mixed, rec_co_design, rec_audit, recommendations_result,
            project_output,
        ],
    )

    pages = [qualitative_page, quantitative_page, usability_page, journey_page, mixed_page, co_design_page, audit_page, recommendations_page]
    buttons = [qualitative_button, quantitative_button, usability_button, journey_button, mixed_methods_button, co_design_button, audit_button, recommendations_button]
    for button, page in zip(buttons, pages):
        button.click(lambda: tuple([gr.update(visible=False)] + [gr.update(visible=False)] * 8), outputs=[project_workspace] + pages)
        button.click(lambda p=page: gr.update(visible=True), outputs=page)

    # Module events
    analyze_button.click(analyze_qualitative, [research_data, selected_project], [qualitative_result, journey_qualitative, project_output])
    quantitative_button_run.click(analyze_quantitative, [quantitative_file, selected_project], [quantitative_result, mm_quantitative, quantitative_gallery, project_output])
    journey_run.click(generate_journey_map, [selected_project, journey_qualitative], [journey_result, project_output])
    usability_run.click(analyze_usability, [selected_project, usability_qualitative, usability_journey], [usability_result, project_output])
    mixed_run.click(mixed_methods_synthesis, [selected_project, mm_qualitative, mm_quantitative, mm_usability, mm_journey], [mixed_result, project_output])
    co_design_run.click(generate_co_design, [selected_project, co_design_brief, co_design_mixed], [co_design_result, project_output])
    audit_run.click(evidence_audit, [selected_project, audit_mixed, audit_co_design, audit_external], [audit_result, project_output])
    recommendations_run.click(generate_recommendations, [selected_project, rec_mixed, rec_co_design, rec_audit], [recommendations_result, project_output])

    # Return navigation for every module.
    for back_button, page in zip([qual_back, quant_back, usability_back, journey_back, mixed_back, co_design_back, audit_back, recommendations_back], pages):
        back_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[project_workspace, page])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
