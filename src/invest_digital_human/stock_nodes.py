from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


QueryKind = Literal["all", "latest", "exists"]


ALIAS_BY_SOURCE = {
    "aapl": {"display": "苹果", "aliases": ["苹果", "apple", "aapl"]},
    "amd": {"display": "AMD", "aliases": ["amd", "advanced micro devices", "超威"]},
    "amzn": {"display": "亚马逊", "aliases": ["亚马逊", "amazon", "amzn"]},
    "avgo": {"display": "博通", "aliases": ["博通", "broadcom", "avgo"]},
    "baba": {"display": "阿里", "aliases": ["阿里", "阿里巴巴", "alibaba", "baba"]},
    "beke": {"display": "贝壳", "aliases": ["贝壳", "ke holdings", "beke"]},
    "bil": {"display": "BIL", "aliases": ["bil"]},
    "brk": {"display": "伯克希尔", "aliases": ["伯克希尔", "巴菲特", "berkshire", "brk", "brk.b", "brk-a", "brk-b"]},
    "costco": {"display": "Costco", "aliases": ["costco", "开市客", "好市多", "cost"]},
    "googl": {"display": "谷歌", "aliases": ["谷歌", "google", "alphabet", "googl", "goog"]},
    "ivv": {"display": "IVV", "aliases": ["ivv"]},
    "jnj": {"display": "强生", "aliases": ["强生", "johnson", "jnj"]},
    "ko": {"display": "可口可乐", "aliases": ["可口可乐", "coca cola", "coca-cola", "ko"]},
    "li": {"display": "理想", "aliases": ["理想", "理想汽车", "li auto", "li"]},
    "lly": {"display": "礼来", "aliases": ["礼来", "eli lilly", "lilly", "lly"]},
    "mcd": {"display": "麦当劳", "aliases": ["麦当劳", "mcdonald", "mcd"]},
    "meta": {"display": "Meta", "aliases": ["meta", "facebook", "脸书"]},
    "mrk": {"display": "默沙东", "aliases": ["默沙东", "merck", "mrk"]},
    "msft": {"display": "微软", "aliases": ["微软", "microsoft", "msft"]},
    "mstr": {"display": "MSTR", "aliases": ["mstr", "microstrategy"]},
    "nflx": {"display": "奈飞", "aliases": ["奈飞", "netflix", "nflx"]},
    "nvda": {"display": "英伟达", "aliases": ["英伟达", "nvidia", "nvda"]},
    "nvo": {"display": "诺和诺德", "aliases": ["诺和诺德", "novo nordisk", "nvo"]},
    "orcl": {"display": "甲骨文", "aliases": ["甲骨文", "oracle", "orcl"]},
    "pdd": {"display": "拼多多", "aliases": ["拼多多", "pdd"]},
    "pg": {"display": "宝洁", "aliases": ["宝洁", "procter", "p&g", "pg"]},
    "pltr": {"display": "PLTR", "aliases": ["pltr", "palantir"]},
    "qqq": {"display": "QQQ", "aliases": ["qqq", "纳指100", "纳斯达克100"]},
    "schd": {"display": "SCHD", "aliases": ["schd"]},
    "shy": {"display": "SHY", "aliases": ["shy"]},
    "spy": {"display": "SPY", "aliases": ["spy", "标普500", "标普"]},
    "tlt": {"display": "TLT", "aliases": ["tlt"]},
    "tsla": {"display": "特斯拉", "aliases": ["特斯拉", "tesla", "tsla"]},
    "tsm": {"display": "台积电", "aliases": ["台积电", "tsmc", "tsm"]},
    "unh": {"display": "联合健康", "aliases": ["联合健康", "联合健康集团", "unitedhealth", "united health", "unh"]},
    "v": {"display": "VISA", "aliases": ["visa", "维萨", "v"]},
    "vig": {"display": "VIG", "aliases": ["vig"]},
    "voo": {"display": "VOO", "aliases": ["voo"]},
    "vym": {"display": "VYM", "aliases": ["vym"]},
    "wmt": {"display": "沃尔玛", "aliases": ["沃尔玛", "walmart", "wmt"]},
}


NODE_INTENT_KEYWORDS = (
    "买入",
    "买点",
    "加仓",
    "建仓",
    "防守",
    "节点",
    "价位",
    "位置",
    "抄底",
    "明确提到",
)
LATEST_KEYWORDS = ("最近一次", "最新", "最近", "最后一次")
EXISTS_KEYWORDS = ("有没有", "是否", "有无", "提到过吗", "有提到")


@dataclass(frozen=True, slots=True)
class StockNodeRecord:
    stock: str
    display_stock: str
    source_key: str
    has_explicit_nodes: bool
    date: str
    article_file: str
    entry_type: str
    nodes: list[str]
    summary: str
    evidence: str


@dataclass(frozen=True, slots=True)
class StockNodeQuery:
    source_key: str
    display_stock: str
    kind: QueryKind
    raw_query: str


class StockNodeKnowledgeBase:
    def __init__(self, master_path: Path) -> None:
        self.master_path = master_path
        self.records = self._load_records(master_path)
        self.records_by_source: dict[str, list[StockNodeRecord]] = {}
        self.empty_sources: set[str] = set()
        for record in self.records:
            if record.has_explicit_nodes:
                self.records_by_source.setdefault(record.source_key, []).append(record)
            else:
                self.empty_sources.add(record.source_key)
        for source_records in self.records_by_source.values():
            source_records.sort(key=lambda item: (item.date, item.article_file), reverse=True)

        self.alias_to_source = self._build_aliases()

    def parse_query(self, query: str) -> StockNodeQuery | None:
        if not self._looks_like_stock_node_query(query):
            return None
        source_key = self._match_source_key(query)
        if source_key is None:
            return None
        return StockNodeQuery(
            source_key=source_key,
            display_stock=self.display_name(source_key),
            kind=self._classify_kind(query),
            raw_query=query,
        )

    def query_all(self, source_key: str) -> list[StockNodeRecord]:
        return list(self.records_by_source.get(source_key, []))

    def query_latest(self, source_key: str) -> StockNodeRecord | None:
        rows = self.records_by_source.get(source_key, [])
        return rows[0] if rows else None

    def has_explicit_nodes(self, source_key: str) -> bool:
        return bool(self.records_by_source.get(source_key))

    def display_name(self, source_key: str) -> str:
        meta = ALIAS_BY_SOURCE.get(source_key)
        return str(meta["display"]) if meta else source_key.upper()

    @staticmethod
    def _load_records(master_path: Path) -> list[StockNodeRecord]:
        records: list[StockNodeRecord] = []
        if not master_path.exists():
            return records
        with master_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                source_key = _source_key_from_json(str(payload.get("source_json", "")))
                records.append(
                    StockNodeRecord(
                        stock=str(payload.get("stock", "")),
                        display_stock=StockNodeKnowledgeBase._display_from_payload(source_key, payload),
                        source_key=source_key,
                        has_explicit_nodes=bool(payload.get("has_explicit_nodes")),
                        date=str(payload.get("date", "")),
                        article_file=str(payload.get("article_file", "")),
                        entry_type=str(payload.get("entry_type", "")),
                        nodes=[str(node) for node in payload.get("nodes", [])],
                        summary=str(payload.get("summary", "")),
                        evidence=str(payload.get("evidence", "")),
                    )
                )
        return records

    @staticmethod
    def _display_from_payload(source_key: str, payload: dict[str, object]) -> str:
        meta = ALIAS_BY_SOURCE.get(source_key)
        if meta:
            return str(meta["display"])
        return str(payload.get("stock", "") or source_key.upper())

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        source_keys = {record.source_key for record in self.records if record.source_key}
        for source_key in source_keys:
            aliases[_normalize_alias(source_key)] = source_key
            meta = ALIAS_BY_SOURCE.get(source_key)
            if not meta:
                continue
            for alias in meta["aliases"]:
                aliases[_normalize_alias(str(alias))] = source_key
        return aliases

    def _match_source_key(self, query: str) -> str | None:
        normalized_query = _normalize_alias(query)
        for alias, source_key in sorted(self.alias_to_source.items(), key=lambda item: len(item[0]), reverse=True):
            if not alias:
                continue
            if _is_short_ascii_alias(alias):
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_query):
                    return source_key
            elif alias in normalized_query:
                return source_key
        return None

    @staticmethod
    def _looks_like_stock_node_query(query: str) -> bool:
        return any(keyword in query for keyword in NODE_INTENT_KEYWORDS)

    @staticmethod
    def _classify_kind(query: str) -> QueryKind:
        if any(keyword in query for keyword in EXISTS_KEYWORDS):
            return "exists"
        if any(keyword in query for keyword in LATEST_KEYWORDS):
            return "latest"
        return "all"


def _source_key_from_json(source_json: str) -> str:
    suffix = "_buy_nodes.json"
    value = source_json.strip().lower()
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def _normalize_alias(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _is_short_ascii_alias(alias: str) -> bool:
    return alias.isascii() and alias.replace(".", "").replace("-", "").isalnum() and len(alias) <= 4
