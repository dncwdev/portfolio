from __future__ import annotations

from typing import Literal, TypeAlias, cast


DomainQueryTemplate: TypeAlias = Literal[
    "적용 법령 및 규정",
    "보안 및 정보보호 요건",
    "안전보건 요건",
    "계약 자격 및 허가 요건",
    "발주 전 준수 요건",
]

DOMAIN_QUERY_ORDER: tuple[DomainQueryTemplate, ...] = (
    "적용 법령 및 규정",
    "보안 및 정보보호 요건",
    "안전보건 요건",
    "계약 자격 및 허가 요건",
    "발주 전 준수 요건",
)

DOMAIN_QUERY_TEMPLATES: dict[DomainQueryTemplate, str] = {
    "적용 법령 및 규정": (
        "구매규격서 작성 단계에서 적용되는 법률, 시행령, 계약예규, 고시, "
        "필수 기재 조항과 법적 근거를 찾는다."
    ),
    "보안 및 정보보호 요건": (
        "구매규격서에 포함해야 하는 보안, 정보보호, 개인정보, 접근통제, "
        "비밀유지, 보안서약 관련 요구사항을 찾는다."
    ),
    "안전보건 요건": (
        "구매규격서에 포함해야 하는 안전보건, 산업안전, 작업자 보호, "
        "위험관리, 안전조치 관련 요구사항을 찾는다."
    ),
    "계약 자격 및 허가 요건": (
        "입찰참가자 또는 계약상대자에게 요구되는 자격, 등록, 면허, 허가, "
        "인증, 실적 요건을 찾는다."
    ),
    "발주 전 준수 요건": (
        "입찰 공고 전 구매규격서 작성 단계에서 확인해야 하는 공정성, 경쟁성, "
        "차별금지, 사전 심의, 필수 첨부조항을 찾는다."
    ),
}


def get_domain_query_templates() -> tuple[DomainQueryTemplate, ...]:
    return DOMAIN_QUERY_ORDER


def normalize_domain_query_templates(
    query_templates: tuple[str, ...] | list[str] | None,
) -> tuple[DomainQueryTemplate, ...]:
    if not query_templates:
        return DOMAIN_QUERY_ORDER

    normalized: list[DomainQueryTemplate] = []
    for query_template in query_templates:
        if query_template not in DOMAIN_QUERY_TEMPLATES:
            raise ValueError(
                f"Unknown domain query template: {query_template}. "
                f"Available: {', '.join(DOMAIN_QUERY_ORDER)}"
            )
        normalized.append(cast(DomainQueryTemplate, query_template))
    return tuple(normalized)


def build_domain_query(
    review_question: str,
    query_template: DomainQueryTemplate,
) -> str:
    normalized_question = review_question.strip()
    query_lines = [
        "검토 단계: 구매규격서 작성 및 입찰 공고 전",
        "검토 초점: 규격서에 법령상 필수 조항이 포함되어 있는지 확인",
    ]
    if normalized_question:
        query_lines.append(f"검토 질문: {normalized_question}")
    query_lines.extend(
        [
            f"도메인 템플릿: {query_template}",
            f"검색 지침: {DOMAIN_QUERY_TEMPLATES[query_template]}",
        ]
    )
    return "\n".join(query_lines)


def format_domain_query_template_guide() -> str:
    return "\n".join(
        f"- {query_template}: {DOMAIN_QUERY_TEMPLATES[query_template]}"
        for query_template in DOMAIN_QUERY_ORDER
    )
