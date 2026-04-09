"""
Document Loader - Task 2 Resume Intelligence Platform

Loads and preprocesses resume documents from:
  1. Kaggle CSV  (UpdatedResumeDataSet.csv from snehaanbhawal/resume-dataset)
  2. Built-in sample data (10 resumes for offline/demo usage)

AC-2 Scalability:
  - Incremental loading: content-hash index prevents re-processing existing docs
  - Supports adding new CSV rows without rebuilding the whole pipeline

Ninja: Corpus update w/o Vector DB rebuild
  - DocumentLoader tracks processed IDs → returns only NEW documents
  - RAGEngine then calls vectorstore.add_documents(new_docs) for incremental update
"""

import os
import csv
import json
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ResumeDocument:
    """A single resume document with metadata."""

    doc_id: str
    category: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "category": self.category,
            "text_preview": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Built-in sample resumes (used when Kaggle CSV is not available)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RESUMES = [
    {
        "category": "Data Science",
        "text": (
            "John Smith | Data Scientist | john.smith@email.com | +1-555-0101\n\n"
            "EXPERIENCE: 5 years in data science and machine learning.\n"
            "Senior Data Scientist at TechCorp (2020-2024): Led ML pipeline development "
            "using Python, TensorFlow, PyTorch, scikit-learn. Built recommendation systems "
            "serving 10M users. A/B testing, statistical analysis, feature engineering.\n\n"
            "SKILLS: Python, R, SQL, TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, "
            "Spark, Hadoop, Docker, Kubernetes, AWS SageMaker, MLflow, Tableau, Power BI\n\n"
            "EDUCATION: M.S. Computer Science, Stanford University, 2019\n\n"
            "CERTIFICATIONS: AWS Machine Learning Specialty, Google Professional Data Engineer"
        ),
    },
    {
        "category": "Data Science",
        "text": (
            "Emily Chen | ML Engineer | emily.chen@email.com | +1-555-0102\n\n"
            "EXPERIENCE: 3 years in ML engineering.\n"
            "ML Engineer at DataCo (2021-2024): Deployed and monitored ML models in production. "
            "Python, machine learning, deep learning with Keras and PyTorch. "
            "Feature engineering, model optimization, real-time inference pipelines.\n\n"
            "SKILLS: Python, PyTorch, Keras, TensorFlow, scikit-learn, SQL, MongoDB, "
            "Docker, Kubernetes, FastAPI, Redis, Kafka, GCP, Vertex AI, Airflow\n\n"
            "EDUCATION: B.S. Mathematics, MIT, 2020\n\n"
            "CERTIFICATIONS: TensorFlow Developer Certificate"
        ),
    },
    {
        "category": "Data Science",
        "text": (
            "Lisa Zhang | Data Analyst | l.zhang@email.com | +1-555-0108\n\n"
            "EXPERIENCE: 4 years in data analysis and business intelligence.\n"
            "Senior Data Analyst at RetailCo (2020-2024): Built Tableau dashboards tracking "
            "KPIs. Python and SQL for data pipeline construction, statistical modeling. "
            "A/B testing for marketing campaigns, customer segmentation.\n\n"
            "SKILLS: Python, R, SQL, Tableau, Power BI, Excel, pandas, NumPy, SciPy, "
            "statsmodels, scikit-learn, PostgreSQL, BigQuery, dbt, Looker, Airflow\n\n"
            "EDUCATION: B.S. Statistics, UCLA, 2019"
        ),
    },
    {
        "category": "Python Developer",
        "text": (
            "Alex Johnson | Python Developer | alex.j@email.com | +1-555-0103\n\n"
            "EXPERIENCE: 4 years in Python software development.\n"
            "Senior Python Developer at WebStartup (2020-2024): Built scalable REST APIs "
            "with Django and FastAPI. PostgreSQL, Redis caching, Celery task queues. "
            "Microservices architecture, CI/CD pipelines, test-driven development.\n\n"
            "SKILLS: Python, Django, FastAPI, Flask, PostgreSQL, MySQL, Redis, Celery, "
            "Docker, Kubernetes, AWS, Git, pytest, SQLAlchemy, GraphQL, REST APIs\n\n"
            "EDUCATION: B.S. Computer Science, University of Washington, 2019"
        ),
    },
    {
        "category": "Python Developer",
        "text": (
            "Maria Rodriguez | Backend Developer | m.rodriguez@email.com | +1-555-0104\n\n"
            "EXPERIENCE: 6 years in backend Python development.\n"
            "Lead Developer at FinTech Inc (2018-2024): Designed high-throughput payment "
            "processing systems using Python, Django REST Framework. PostgreSQL optimization, "
            "Redis, message queues with RabbitMQ. 99.99% uptime SLAs.\n\n"
            "SKILLS: Python, Django, DRF, PostgreSQL, Redis, RabbitMQ, Docker, "
            "Kubernetes, AWS, Terraform, Prometheus, Grafana, Git, SQLAlchemy\n\n"
            "EDUCATION: M.S. Software Engineering, Carnegie Mellon, 2018\n\n"
            "CERTIFICATIONS: AWS Solutions Architect Associate"
        ),
    },
    {
        "category": "Java Developer",
        "text": (
            "David Kim | Java Developer | d.kim@email.com | +1-555-0105\n\n"
            "EXPERIENCE: 7 years in Java enterprise development.\n"
            "Senior Java Developer at EnterpriseBank (2017-2024): Built microservices with "
            "Spring Boot, Spring Cloud. Oracle DB, Apache Kafka, RESTful APIs. "
            "Performance optimization, JVM tuning, security compliance (PCI-DSS).\n\n"
            "SKILLS: Java, Spring Boot, Spring MVC, Hibernate, Oracle, MySQL, PostgreSQL, "
            "Apache Kafka, Redis, Docker, Kubernetes, Jenkins, Maven, Gradle, JUnit, Mockito\n\n"
            "EDUCATION: B.S. Computer Science, UC Berkeley, 2017\n\n"
            "CERTIFICATIONS: Oracle Certified Professional Java SE Developer"
        ),
    },
    {
        "category": "DevOps Engineer",
        "text": (
            "Sarah Williams | DevOps Engineer | s.williams@email.com | +1-555-0106\n\n"
            "EXPERIENCE: 5 years in DevOps and cloud infrastructure.\n"
            "DevOps Lead at CloudCo (2019-2024): Managed CI/CD pipelines, Kubernetes clusters "
            "on AWS EKS. Terraform infrastructure as code, monitoring with Prometheus/Grafana. "
            "Security hardening, disaster recovery, 99.9% uptime SLAs.\n\n"
            "SKILLS: Kubernetes, Docker, Terraform, Ansible, Jenkins, GitLab CI, AWS, GCP, "
            "Azure, Prometheus, Grafana, ELK Stack, Python, Bash, Linux, Helm\n\n"
            "EDUCATION: B.S. Information Technology, Arizona State University, 2018\n\n"
            "CERTIFICATIONS: CKA (Certified Kubernetes Administrator), AWS DevOps Professional"
        ),
    },
    {
        "category": "Network Security Engineer",
        "text": (
            "Robert Brown | Security Engineer | r.brown@email.com | +1-555-0107\n\n"
            "EXPERIENCE: 8 years in network security and cybersecurity.\n"
            "Security Architect at SecureCorp (2016-2024): Designed zero-trust network "
            "architecture. Implemented IDS/IPS, SIEM solutions, penetration testing. "
            "Compliance with PCI-DSS, SOC2, ISO 27001. Security awareness training.\n\n"
            "SKILLS: Network Security, Firewall Management, IDS/IPS, SIEM, Penetration Testing, "
            "Python, Bash, Wireshark, Nessus, Splunk, AWS Security, Zero Trust, CISSP\n\n"
            "EDUCATION: B.S. Network Engineering, Purdue University, 2015\n\n"
            "CERTIFICATIONS: CISSP, CEH, CompTIA Security+, AWS Security Specialty"
        ),
    },
    {
        "category": "Web Designing",
        "text": (
            "Tom Anderson | Full Stack Developer | t.anderson@email.com | +1-555-0109\n\n"
            "EXPERIENCE: 5 years in web development.\n"
            "Full Stack Developer at AgencyX (2019-2024): Built responsive web applications "
            "using React, Vue.js, Node.js. REST and GraphQL APIs, PostgreSQL, MongoDB. "
            "Performance optimization, accessibility compliance (WCAG 2.1), PWA development.\n\n"
            "SKILLS: JavaScript, TypeScript, React, Vue.js, Angular, Node.js, Express, "
            "HTML5, CSS3, PostgreSQL, MongoDB, GraphQL, REST APIs, Docker, AWS, Git\n\n"
            "EDUCATION: B.S. Computer Science, Georgia Tech, 2018"
        ),
    },
    {
        "category": "Business Analyst",
        "text": (
            "Patricia Moore | Business Analyst | p.moore@email.com | +1-555-0110\n\n"
            "EXPERIENCE: 6 years in business analysis and project management.\n"
            "Senior BA at ConsultFirm (2018-2024): Requirements gathering, user story writing, "
            "process mapping, stakeholder management. Agile/Scrum facilitation, sprint planning. "
            "SQL for data analysis, Tableau dashboards for executive reporting.\n\n"
            "SKILLS: Business Analysis, Requirements Gathering, User Stories, Process Mapping, "
            "SQL, Tableau, Excel, JIRA, Confluence, Power BI, Agile, Scrum, Stakeholder Mgmt\n\n"
            "EDUCATION: MBA, Northwestern University, 2018\n\n"
            "CERTIFICATIONS: CBAP (Certified Business Analysis Professional), PMI-ACP"
        ),
    },
]


class DocumentLoader:
    """
    Loads resume documents from CSV or built-in sample data.

    AC-2: Incremental loading skips already-indexed documents so adding
          new resumes to an existing corpus does not re-process old ones.
    """

    def __init__(self, index_path: str = ""):
        self._index_path = index_path
        self._processed_ids: set = self._load_index()

    # ── Index persistence ─────────────────────────────────────────────────────

    def _load_index(self) -> set:
        if self._index_path and os.path.exists(self._index_path):
            with open(self._index_path, encoding="utf-8") as fh:
                return set(json.load(fh))
        return set()

    def _save_index(self) -> None:
        if not self._index_path:
            return
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        with open(self._index_path, "w", encoding="utf-8") as fh:
            json.dump(sorted(self._processed_ids), fh)

    @staticmethod
    def _make_doc_id(category: str, text: str) -> str:
        """Deterministic content-hash ID (first 12 hex chars of MD5)."""
        content = f"{category}:{text[:200]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # ── Public loaders ────────────────────────────────────────────────────────

    def load_sample_data(self) -> List[ResumeDocument]:
        """Return the 10 built-in sample resumes (no external files needed)."""
        docs = []
        for i, sample in enumerate(SAMPLE_RESUMES):
            doc_id = self._make_doc_id(sample["category"], sample["text"])
            docs.append(
                ResumeDocument(
                    doc_id=doc_id,
                    category=sample["category"],
                    text=sample["text"],
                    metadata={
                        "category": sample["category"],
                        "doc_id": doc_id,
                        "source": "sample",
                        "row_index": i,
                        "text_length": len(sample["text"]),
                    },
                )
            )
            self._processed_ids.add(doc_id)
        self._save_index()
        return docs

    def load_from_csv(
        self,
        csv_path: str,
        max_docs: Optional[int] = None,
        incremental: bool = True,
    ) -> List[ResumeDocument]:
        """
        Load resumes from Kaggle CSV (UpdatedResumeDataSet.csv).

        Args:
            csv_path:    Path to the CSV file.
            max_docs:    Cap total documents loaded (None = all).
            incremental: Skip docs already in the processed-ID index.

        Returns:
            List of *new* ResumeDocument objects only.
        """
        docs: List[ResumeDocument] = []

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if max_docs is not None and len(docs) >= max_docs:
                    break

                category = row.get("Category", "Unknown").strip()
                # Support multiple known schemas:
                # - UpdatedResumeDataSet.csv : Resume
                # - Resume.csv               : Resume_str / Resume_html
                text = (
                    row.get("Resume", "")
                    or row.get("Resume_str", "")
                    or row.get("Resume_html", "")
                )
                text = self._normalize_resume_text(text)
                if not text:
                    continue

                doc_id = self._make_doc_id(category, text)

                if incremental and doc_id in self._processed_ids:
                    continue  # Skip already-indexed document

                docs.append(
                    ResumeDocument(
                        doc_id=doc_id,
                        category=category,
                        text=text,
                        metadata={
                            "category": category,
                            "doc_id": doc_id,
                            "source": "kaggle_csv",
                            "row_index": i,
                            "text_length": len(text),
                        },
                    )
                )
                self._processed_ids.add(doc_id)

        self._save_index()
        return docs

    @staticmethod
    def _normalize_resume_text(text: str) -> str:
        """Normalize raw resume text and strip basic HTML if needed."""
        if not text:
            return ""

        # Remove common HTML tags when Resume_html is used.
        if "<" in text and ">" in text:
            text = re.sub(r"<[^>]+>", " ", text)

        # Collapse extra whitespace.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def chunk_documents(
        documents: List[ResumeDocument],
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> List[Dict]:
        """
        Split each resume into overlapping text chunks for embedding.

        Returns dicts compatible with LangChain Document format:
          {"page_content": str, "metadata": dict}
        """
        chunks: List[Dict] = []
        for doc in documents:
            text = doc.text
            start, chunk_idx = 0, 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(
                    {
                        "page_content": text[start:end],
                        "metadata": {
                            **doc.metadata,
                            "chunk_idx": chunk_idx,
                            "chunk_id": f"{doc.doc_id}_c{chunk_idx}",
                        },
                    }
                )
                chunk_idx += 1
                if end == len(text):
                    break
                start += chunk_size - chunk_overlap
        return chunks

    @staticmethod
    def get_category_stats(documents: List[ResumeDocument]) -> Dict[str, int]:
        """Count documents per category."""
        stats: Dict[str, int] = {}
        for doc in documents:
            stats[doc.category] = stats.get(doc.category, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
