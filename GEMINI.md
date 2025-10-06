# GEMINI Code Analysis: wal-fact-checker

## 1. Project Overview

**`wal-fact-checker`** is a Python-based, multi-agent system for automated fact-checking. It leverages the Google Agent Development Kit (ADK) and Gemini large language models to verify factual claims by breaking them down, conducting web research, and synthesizing the findings into a structured report.

### 1.1. Core Features

-   **Multi-Agent Architecture:** A modular pipeline of specialized agents for each stage of the fact-checking process.
-   **Automated Research:** Utilizes custom tools for web searches and content scraping to gather evidence.
-   **Confidence and Priority:** Incorporates confidence scoring for claims and priority levels for research questions to guide the workflow.
-   **Dynamic Resource Allocation:** Adjusts the number of tool calls based on the priority of the research question.
-   **Scalable Deployment:** Containerized with Docker for deployment on Google Cloud Run.
-   **Observability:** Integrated with Langfuse for tracing and monitoring.

## 2. System Architecture

The system is designed as a sequential pipeline of agents, orchestrated by the `FactCheckOrchestrator`. This modular design ensures a clear separation of concerns.

### 2.1. Fact-Checking Workflow

The workflow is divided into three main stages:

1.  **Analysis & Strategy (`AnalysisStage`)**: Deconstructs the initial input and creates a research plan.
2.  **Research (`ResearchStage`)**: Executes the research plan by gathering evidence from the web.
3.  **Synthesis & Verification (`SynthesisStage`)**: Adjudicates the evidence and generates the final report.

### 2.2. Data Flow

1.  **Input:** The user provides a claim to be fact-checked.
2.  **`ClaimStructuringAgent`:** The input text is broken down into atomic, verifiable claims, each with a `confidence` score.
3.  **`GapIdentificationAgent`:** The structured claims are analyzed to generate a list of research questions, each with a `priority`.
4.  **`ResearchOrchestratorAgent`:** The research questions are researched in order of priority.
5.  **`SingleQuestionResearchAgent`:** For each question, this agent uses search and scrape tools to find answers. The number of tool calls is determined by the question's priority.
6.  **`EvidenceAdjudicatorAgent`:** The gathered evidence is used to adjudicate each claim, resulting in a verdict (`True`, `False`, or `Could Not Be Verified`).
7.  **`ReportTransformationAgent`:** The final adjudicated report is transformed into a user-friendly JSON format.

### 2.3. Technology Stack

-   **Backend:** FastAPI
-   **Agent Framework:** Google Agent Development Kit (ADK)
-   **Language Models:** Google Gemini
-   **Package Management:** uv
-   **Containerization:** Docker
-   **Deployment:** Google Cloud Run
-   **Observability:** Langfuse

## 3. Key Agents

### 3.1. `ClaimStructuringAgent`

-   **Purpose:** To deconstruct complex statements into simple, verifiable claims.
-   **Output:** A list of claims, each with a `confidence` score.

### 3.2. `GapIdentificationAgent`

-   **Purpose:** To generate targeted research questions from the structured claims.
-   **Output:** A list of research questions, each with a `priority`.

### 3.3. `SingleQuestionResearchAgent`

-   **Purpose:** To answer a single research question using web search and scraping.
-   **Logic:** The number of search and scrape tool calls is dynamically adjusted based on the question's priority.

### 3.4. `EvidenceAdjudicatorAgent`

-   **Purpose:** To synthesize evidence and produce a final verdict for each claim.
-   **Output:** A structured report with verdicts and supporting evidence.

## 4. How to Run

### 4.1. Local Development

1.  **Install Dependencies:**
    ```bash
    uv sync --dev
    ```

2.  **Run the Application:**
    ```bash
    docker-compose up
    ```

### 4.2. Deployment

The project includes a GitHub Actions workflow for continuous deployment to Google Cloud Run.