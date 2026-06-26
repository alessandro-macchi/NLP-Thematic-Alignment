"""Data loading and saving utilities for journal alignment."""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from journal_alignment.preprocessing import normalize_whitespace


PUBMED_COLUMNS = [
    "pmid",
    "title",
    "abstract",
    "year",
    "journal",
    "doi",
    "publication_date",
    "authors",
    "keywords",
]


class PubMedClient:
    """Small PubMed client using NCBI E-utilities and the standard library."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        tool: str = "journal-alignment-project",
        rate_limit_seconds: float = 0.34,
    ) -> None:
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.rate_limit_seconds = rate_limit_seconds

    def search_articles(
        self,
        journal_name: str,
        start_year: int,
        end_year: int,
        max_results: int = 200,
    ) -> list[str]:
        """Return PubMed IDs for a journal and publication-year range."""

        if not journal_name.strip():
            raise ValueError("journal_name must not be empty.")
        if start_year > end_year:
            raise ValueError("start_year must be earlier than or equal to end_year.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        params = {
            "db": "pubmed",
            "term": f'"{journal_name}"[Journal]',
            "retmax": str(max_results),
            "retmode": "xml",
            "sort": "pub date",
            "datetype": "pdat",
            "mindate": str(start_year),
            "maxdate": str(end_year),
        }
        xml_bytes = self._request(self.ESEARCH_URL, params)
        root = ET.fromstring(xml_bytes)
        error = root.findtext(".//ERROR")
        if error:
            raise RuntimeError(f"PubMed search failed: {error}")
        return [element.text for element in root.findall(".//IdList/Id") if element.text]

    def fetch_articles(
        self,
        pmids: list[str],
    ) -> pd.DataFrame:
        """Fetch PubMed records for ``pmids`` and return article metadata."""

        if not pmids:
            return pd.DataFrame(columns=PUBMED_COLUMNS)

        frames = []
        for batch in _batched(pmids, batch_size=100):
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }
            xml_bytes = self._request(self.EFETCH_URL, params)
            frames.append(_parse_pubmed_xml(xml_bytes))

        if not frames:
            return pd.DataFrame(columns=PUBMED_COLUMNS)
        return pd.concat(frames, ignore_index=True)[PUBMED_COLUMNS]

    def fetch_journal_articles_last_10_years(
        self,
        journal_name: str,
        max_results: int = 200,
    ) -> pd.DataFrame:
        """Fetch article metadata for the last 10 years up to the current year."""

        end_year = date.today().year
        start_year = end_year - 10
        pmids = self.search_articles(
            journal_name=journal_name,
            start_year=start_year,
            end_year=end_year,
            max_results=max_results,
        )
        return self.fetch_articles(pmids)

    def _request(self, url: str, params: dict[str, str]) -> bytes:
        request_params = {
            **params,
            "tool": self.tool,
        }
        if self.email:
            request_params["email"] = self.email
        if self.api_key:
            request_params["api_key"] = self.api_key

        encoded_params = urllib.parse.urlencode(request_params)
        request_url = f"{url}?{encoded_params}"

        try:
            with urllib.request.urlopen(request_url, timeout=30) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"PubMed request failed: {exc}") from exc
        finally:
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)


class ArticleDataset:
    """Load, validate, and save article data plus a manual Aims & Scope file."""

    REQUIRED_COLUMNS = {"title", "abstract", "year"}

    def __init__(
        self,
        articles_path: str | Path,
        aims_scope_path: str | Path,
        min_abstract_length: int = 50,
    ) -> None:
        self.articles_path = Path(articles_path)
        self.aims_scope_path = Path(aims_scope_path)
        self.min_abstract_length = min_abstract_length

    def load_articles(self) -> pd.DataFrame:
        """Load the raw article CSV retrieved from PubMed."""

        if not self.articles_path.exists():
            raise FileNotFoundError(
                f"Article CSV file does not exist: {self.articles_path}"
            )
        return pd.read_csv(self.articles_path)

    def load_aims_scope(self) -> str:
        """Load a manually provided Aims & Scope text file."""

        if not self.aims_scope_path.exists():
            raise FileNotFoundError(
                f"Aims & Scope TXT file does not exist: {self.aims_scope_path}"
            )
        aims_scope = self.aims_scope_path.read_text(encoding="utf-8").strip()
        if not aims_scope:
            raise ValueError(
                f"Aims & Scope TXT file is empty: {self.aims_scope_path}"
            )
        return aims_scope

    def validate_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean article rows needed by later project steps."""

        missing_columns = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Article data is missing required columns: {missing}")

        clean_df = df.copy()
        for column in ["title", "abstract"]:
            clean_df[column] = clean_df[column].astype("string").map(
                normalize_whitespace
            )
            clean_df.loc[clean_df[column] == "", column] = pd.NA

        clean_df["year"] = pd.to_numeric(clean_df["year"], errors="coerce")
        clean_df = clean_df.dropna(subset=["title", "abstract", "year"])
        clean_df = clean_df[
            clean_df["abstract"].str.len() >= self.min_abstract_length
        ].copy()
        clean_df["year"] = clean_df["year"].astype(int)
        clean_df = clean_df.drop_duplicates(subset=["title", "abstract"])
        return clean_df.reset_index(drop=True)

    def save_processed(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Save a processed article DataFrame as CSV."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _parse_pubmed_xml(xml_bytes: bytes | str) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    records = []
    for article in root.findall(".//PubmedArticle"):
        records.append(_parse_pubmed_article(article))
    return pd.DataFrame(records, columns=PUBMED_COLUMNS)


def _parse_pubmed_article(article: ET.Element) -> dict[str, object]:
    citation = article.find("MedlineCitation")
    article_node = citation.find("Article") if citation is not None else None

    abstract_parts = []
    if article_node is not None:
        for abstract_text in article_node.findall("./Abstract/AbstractText"):
            label = abstract_text.attrib.get("Label")
            text = _element_text(abstract_text)
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)

    return {
        "pmid": _first_text(article, "./MedlineCitation/PMID"),
        "title": _element_text(article_node.find("ArticleTitle"))
        if article_node is not None and article_node.find("ArticleTitle") is not None
        else "",
        "abstract": normalize_whitespace(" ".join(abstract_parts)),
        "year": _extract_year(article),
        "journal": _journal_name(article_node),
        "doi": _article_id(article, "doi"),
        "publication_date": _publication_date(article),
        "authors": _authors(article_node),
        "keywords": _keywords(citation),
    }


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_whitespace(" ".join(element.itertext()))


def _first_text(element: ET.Element, path: str) -> str:
    return normalize_whitespace(element.findtext(path) or "")


def _journal_name(article_node: ET.Element | None) -> str:
    if article_node is None:
        return ""
    title = _first_text(article_node, "./Journal/Title")
    if title:
        return title
    return _first_text(article_node, "./Journal/ISOAbbreviation")


def _article_id(article: ET.Element, id_type: str) -> str:
    for article_id in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == id_type:
            return _element_text(article_id)
    return ""


def _extract_year(article: ET.Element) -> int | None:
    year_paths = [
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
        "./MedlineCitation/Article/ArticleDate/Year",
        "./PubmedData/History/PubMedPubDate/Year",
    ]
    for path in year_paths:
        year = _year_from_text(article.findtext(path))
        if year is not None:
            return year

    medline_date = article.findtext(
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate"
    )
    return _year_from_text(medline_date)


def _publication_date(article: ET.Element) -> str:
    pub_date = article.find(
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate"
    )
    if pub_date is None:
        return ""

    year = _year_from_text(pub_date.findtext("Year")) or _year_from_text(
        pub_date.findtext("MedlineDate")
    )
    if year is None:
        return ""

    month = _month_number(pub_date.findtext("Month"))
    day = _day_number(pub_date.findtext("Day"))

    if month and day:
        return f"{year}-{month:02d}-{day:02d}"
    if month:
        return f"{year}-{month:02d}"
    return str(year)


def _year_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\b(18|19|20|21)\d{2}\b", text)
    if match:
        return int(match.group(0))
    return None


def _month_number(text: str | None) -> int | None:
    if not text:
        return None
    text = text.strip()
    if text.isdigit():
        month = int(text)
        return month if 1 <= month <= 12 else None

    month_lookup = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    return month_lookup.get(text.lower())


def _day_number(text: str | None) -> int | None:
    if not text or not text.strip().isdigit():
        return None
    day = int(text.strip())
    return day if 1 <= day <= 31 else None


def _authors(article_node: ET.Element | None) -> str:
    if article_node is None:
        return ""

    names = []
    for author in article_node.findall("./AuthorList/Author"):
        collective_name = _first_text(author, "CollectiveName")
        if collective_name:
            names.append(collective_name)
            continue

        last_name = _first_text(author, "LastName")
        fore_name = _first_text(author, "ForeName")
        initials = _first_text(author, "Initials")
        given_name = fore_name or initials
        if last_name and given_name:
            names.append(f"{fore_name or initials} {last_name}")
        elif last_name:
            names.append(last_name)

    return "; ".join(names)


def _keywords(citation: ET.Element | None) -> str:
    if citation is None:
        return ""
    keywords = [_element_text(keyword) for keyword in citation.findall(".//Keyword")]
    return "; ".join(keyword for keyword in keywords if keyword)
