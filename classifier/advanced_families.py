"""Shared bilingual gates for advanced mathematical problem families."""

from __future__ import annotations

import re


DIRECTED_EULER_CIRCUIT_PATTERN = (
    r"BEST\s*(?:定理|theorem)|"
    r"(?:有向(?:多重)?图|有向边|弧序列|弧)[\s\S]{0,600}"
    r"(?:欧拉(?:回路|闭迹|巡游|环游)|每条弧[^。！？\n]{0,50}(?:恰好|正好)?一次)|"
    r"(?:欧拉(?:回路|闭迹|巡游|环游)|每条弧[^。！？\n]{0,50}(?:恰好|正好)?一次)"
    r"[\s\S]{0,600}(?:有向(?:多重)?图|有向边|弧序列|弧)|"
    r"\b(?:directed\s+(?:multi)?graphs?|digraphs?|arcs?)\b[\s\S]{0,600}"
    r"\b(?:euler(?:ian)?\s+(?:circuits?|tours?|trails?)|"
    r"closed\s+euler(?:ian)?\s+(?:trails?|walks?))\b|"
    r"\b(?:euler(?:ian)?\s+(?:circuits?|tours?|trails?)|"
    r"closed\s+euler(?:ian)?\s+(?:trails?|walks?))\b[\s\S]{0,600}"
    r"\b(?:directed\s+(?:multi)?graphs?|digraphs?|arcs?)\b"
)

PLANE_ROOTED_TREE_PATTERN = (
    r"有序平面(?:有根|根)树|平面有序(?:有根|根)树|有序(?:有根|根)平面树|"
    r"平面(?:有根|根)树[^。！？\n]{0,160}(?:出度|度数分布|度序列|计数|数量)|"
    r"Łukasiewicz|Lukasiewicz|卢卡西维茨|Łukasiewicz\s*词|"
    r"\b(?:ordered\s+plane\s+(?:rooted\s+)?trees?|"
    r"plane\s+ordered\s+(?:rooted\s+)?trees?|rooted\s+plane\s+trees?)\b|"
    r"\b(?:out[- ]?degree\s+(?:profile|distribution|sequence)|degree\s+profile)\b"
    r"[\s\S]{0,260}\b(?:plane|ordered|rooted)\s+trees?\b|"
    r"\b(?:plane|ordered|rooted)\s+trees?\b[\s\S]{0,260}"
    r"\b(?:out[- ]?degree\s+(?:profile|distribution|sequence)|degree\s+profile)\b"
)

LACUNARY_NATURAL_BOUNDARY_PATTERN = (
    r"自然边界|稀疏幂级数|幂级数间隙|Fabry\s*间隙|Hadamard\s*间隙|"
    r"(?:幂级数|级数)[\s\S]{0,900}(?:解析延拓|全纯延拓)[\s\S]{0,220}"
    r"(?:每(?:一|条|个)?(?:边界)?圆?弧|任意(?:边界)?圆?弧|整个(?:收敛)?圆周)|"
    r"(?:解析延拓|全纯延拓)[\s\S]{0,220}"
    r"(?:每(?:一|条|个)?(?:边界)?圆?弧|任意(?:边界)?圆?弧|整个(?:收敛)?圆周)|"
    r"\b(?:natural\s+boundar(?:y|ies)|lacunary\s+(?:power\s+)?series|"
    r"fabry(?:'s)?\s+gap(?:\s+theorem)?|hadamard(?:'s)?\s+gap(?:\s+theorem)?)\b|"
    r"\b(?:power\s+series|series)\b[\s\S]{0,900}\banalytic\s+continuation\b"
    r"[\s\S]{0,220}\b(?:every|each|any)\s+(?:boundary\s+)?arc\b|"
    r"\banalytic\s+continuation\b[\s\S]{0,220}"
    r"\b(?:every|each|any)\s+(?:boundary\s+)?arc\b"
)

RUNGE_KUTTA_STABILITY_PATTERN = (
    r"(?:龙格[-－— ]?库塔|Runge[- ]Kutta|Butcher\s*(?:表|tableau)|"
    r"\b(?:DIRK|SDIRK|ESDIRK)\b)[\s\S]{0,650}"
    r"(?:稳定函数|绝对稳定|A[- ]?稳定|L[- ]?稳定|稳定域|"
    r"stability\s+function|absolute\s+stability|A[- ]?stabl\w*|L[- ]?stabl\w*)|"
    r"(?:稳定函数|绝对稳定|A[- ]?稳定|L[- ]?稳定|稳定域|"
    r"stability\s+function|absolute\s+stability|A[- ]?stabl\w*|L[- ]?stabl\w*)"
    r"[\s\S]{0,650}(?:龙格[-－— ]?库塔|Runge[- ]Kutta|Butcher\s*(?:表|tableau)|"
    r"\b(?:DIRK|SDIRK|ESDIRK)\b)"
)

SPHERICAL_TRIANGLE_AREA_PATTERN = (
    r"Girard\s*(?:定理|公式|theorem|formula)|球面超额|球面盈|"
    r"球面三角形[\s\S]{0,500}(?:面积|球面余弦定理)|"
    r"(?:面积|球面余弦定理)[\s\S]{0,500}球面三角形|"
    r"\bspherical\s+(?:triangle|excess)\b[\s\S]{0,500}"
    r"\b(?:area|law\s+of\s+cosines?|cosine\s+law)\b|"
    r"\b(?:area|law\s+of\s+cosines?|cosine\s+law)\b[\s\S]{0,500}"
    r"\bspherical\s+triangle\b|"
    r"\bgeodesic\s+triangle\b[\s\S]{0,500}\b(?:sphere|spherical)\b"
    r"[\s\S]{0,240}\barea\b|"
    r"\barea\b[\s\S]{0,240}\bgeodesic\s+triangle\b"
    r"[\s\S]{0,500}\b(?:sphere|spherical)\b"
)

WEIERSTRASS_SINE_PRODUCT_PATTERN = (
    r"Weierstrass\s*(?:正弦)?乘积|正弦函数[^。！？\n]{0,180}(?:无穷乘积|无限乘积)|"
    r"双曲正弦[^。！？\n]{0,180}(?:无穷乘积|无限乘积)|"
    r"(?:无穷乘积|无限乘积)[^。！？\n]{0,180}(?:正弦函数|双曲正弦)|"
    r"\bweierstrass(?:'s)?\s+(?:sine\s+)?product\b|"
    r"\b(?:infinite\s+product|product\s+formula)\b[\s\S]{0,260}"
    r"\b(?:sin(?:e)?|sinh|hyperbolic\s+sine)\b|"
    r"\b(?:sin(?:e)?|sinh|hyperbolic\s+sine)\b[\s\S]{0,260}"
    r"\b(?:infinite\s+product|product\s+formula)\b"
)

TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN = (
    r"(?=[\s\S]{0,1400}(?:二维|平面上|"
    r"\\mathbb\s*\{?R\}?\s*\^\s*\{?2\}?|"
    r"\bR\s*\^\s*\{?2\}?\b|two[- ]dimensional|"
    r"\bin\s+(?:the\s+)?plane\b|\bon\s+(?:the\s+)?plane\b))"
    r"(?=[\s\S]{0,1400}(?:双调和|多调和|"
    r"\\Delta\s*\^\s*\{?(?:[2-9]\d*|m)\}?|"
    r"\b(?:bi|poly)harmonic\b))"
    r"(?=[\s\S]{0,1400}(?:基本解|基解|"
    r"\bfundamental\s+solutions?\b|\bgreen(?:'s)?\s+functions?\b))"
)


SPECIALIZED_TOPIC_PATTERNS = (
    ("directed_euler_circuits", DIRECTED_EULER_CIRCUIT_PATTERN),
    ("plane_rooted_tree_enumeration", PLANE_ROOTED_TREE_PATTERN),
    ("lacunary_natural_boundary", LACUNARY_NATURAL_BOUNDARY_PATTERN),
    ("runge_kutta_stability", RUNGE_KUTTA_STABILITY_PATTERN),
    ("spherical_triangle_area", SPHERICAL_TRIANGLE_AREA_PATTERN),
    ("weierstrass_sine_product", WEIERSTRASS_SINE_PRODUCT_PATTERN),
    (
        "two_dimensional_polyharmonic_fundamental_solution",
        TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
    ),
)

SPECIALIZED_TOPICS = frozenset(topic for topic, _ in SPECIALIZED_TOPIC_PATTERNS)


def matches(pattern: str, text: str) -> bool:
    """Return whether a shared family gate matches public problem text."""
    return bool(re.search(pattern, str(text or ""), re.IGNORECASE | re.DOTALL))
