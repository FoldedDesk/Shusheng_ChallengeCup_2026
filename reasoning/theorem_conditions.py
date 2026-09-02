"""Fail-closed theorem-hypothesis protocols derived from the current statement.

These contracts do not decide whether a theorem applies and do not certify an
answer.  They only make the theorem's standard admission obligations explicit
when the statement itself names that theorem or method.  This avoids both a
generic critic call and silently selecting a theorem from a remembered source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TheoremConditionContract:
    name: str
    conditions: tuple[str, ...]
    conclusion_check: str

    def render(self, *, review: bool) -> str:
        action = "Audit" if review else "Before applying"
        numbered = " ".join(
            f"H{index}: {condition}"
            for index, condition in enumerate(self.conditions, start=1)
        )
        return (
            f"Theorem condition contract ({self.name}, untrusted routing aid). "
            f"{action} the theorem, establish from the current statement: {numbered} "
            f"Conclusion check: {self.conclusion_check} "
            "If any hypothesis is unavailable, do not cite the theorem; use a different "
            "argument or state that the proposed interchange is not justified."
        )


_CONTRACTS = {
    "dominated convergence theorem": TheoremConditionContract(
        "dominated convergence theorem",
        (
            "the functions and limit are measurable",
            "the sequence converges pointwise almost everywhere",
            "one integrable dominator independent of the sequence index bounds every term almost everywhere",
        ),
        "state the justified limit-integral interchange and retain the almost-everywhere scope",
    ),
    "monotone convergence theorem": TheoremConditionContract(
        "monotone convergence theorem",
        (
            "every term is measurable and nonnegative",
            "the sequence is monotone increasing almost everywhere",
            "the pointwise almost-everywhere limit is identified",
        ),
        "the integrals increase to the integral of the limit, possibly with value infinity",
    ),
    "Fatou lemma": TheoremConditionContract(
        "Fatou lemma",
        (
            "the sequence is measurable",
            "the nonnegativity hypothesis, or a valid common integrable lower-bound reduction, is established",
        ),
        "preserve the direction of the liminf inequality",
    ),
    "Fubini theorem": TheoremConditionContract(
        "Fubini theorem",
        (
            "the product-measure setting and measurability are established",
            "absolute integrability over the product space is established",
        ),
        "both iterated integrals exist in the required finite sense and equal the product-space integral",
    ),
    "Tonelli theorem": TheoremConditionContract(
        "Tonelli theorem",
        (
            "the product-measure setting and measurability are established",
            "the integrand is nonnegative",
        ),
        "allow extended values and preserve the nonnegative-integral conclusion",
    ),
    "Cauchy integral formula": TheoremConditionContract(
        "Cauchy integral formula",
        (
            "the function is holomorphic on an open set containing the contour and its interior",
            "the evaluation point lies inside and no singularity lies on the contour",
            "the contour orientation or winding number is accounted for",
        ),
        "include the correct winding/orientation factor and every requested derivative order",
    ),
    "residue theorem": TheoremConditionContract(
        "residue theorem",
        (
            "all isolated singularities relevant to the contour are located and classified",
            "no singularity lies on the contour",
            "the enclosed poles, contour orientation, and winding numbers are accounted for",
        ),
        "sum exactly the enclosed residues with the correct orientation factor",
    ),
    "Radon-Nikodym theorem": TheoremConditionContract(
        "Radon-Nikodym theorem",
        (
            "the two measures and the direction of absolute continuity are identified",
            "the required sigma-finiteness or the stronger finite-measure hypotheses are established",
            "the derivative is measurable and is asserted only up to almost-everywhere equality",
        ),
        "verify the integral representation against every measurable set and preserve uniqueness only almost everywhere",
    ),
    "Gauss-Bonnet theorem": TheoremConditionContract(
        "Gauss-Bonnet theorem",
        (
            "the surface, metric, compact region, and orientation are fixed",
            "the boundary regularity, induced orientation, geodesic-curvature convention, and any corner angles are accounted for",
        ),
        "include the Gaussian-curvature, boundary, and corner terms with conventions consistent with the Euler characteristic",
    ),
    "Sylow theorem": TheoremConditionContract(
        "Sylow theorem",
        (
            "the group is finite and p is prime",
            "the exact p-power dividing the group order is identified",
            "existence, conjugacy, and subgroup-count congruence statements are not interchanged",
        ),
        "check both divisibility of the Sylow-subgroup count and its congruence modulo p",
    ),
    "isomorphism theorem": TheoremConditionContract(
        "isomorphism theorem",
        (
            "the stated map is a homomorphism with the correct domain and codomain",
            "its kernel and image are computed and the quotient is well-defined",
        ),
        "construct the induced map and verify that it is well-defined, injective, and onto the stated image",
    ),
    "Hahn-Banach theorem": TheoremConditionContract(
        "Hahn-Banach theorem",
        (
            "the scalar field and the real or complex form of the theorem are identified",
            "the starting functional is linear on the stated subspace and is dominated or bounded as required",
            "the proposed extension preserves the required domination or norm",
        ),
        "state the extension on the whole requested space without claiming uniqueness unless separately proved",
    ),
    "open mapping theorem": TheoremConditionContract(
        "open mapping theorem",
        (
            "both domain and codomain are Banach spaces",
            "the operator is linear, bounded, and surjective",
        ),
        "derive openness or the equivalent bounded-inverse consequence without assuming injectivity unless it is given",
    ),
    "closed graph theorem": TheoremConditionContract(
        "closed graph theorem",
        (
            "the domain and codomain are Banach spaces",
            "the operator is linear and defined on the whole stated domain",
            "its graph is actually closed in the product topology",
        ),
        "conclude boundedness only for the operator covered by those hypotheses",
    ),
    "central limit theorem": TheoremConditionContract(
        "central limit theorem",
        (
            "the independence or dependence assumptions match the named version",
            "the centering, variance, and required finite-moment hypotheses are established",
            "the normalization uses the correct sample size and variance scale",
        ),
        "state convergence in distribution to the correctly standardized normal law",
    ),
    "law of large numbers": TheoremConditionContract(
        "law of large numbers",
        (
            "the random variables satisfy the independence and identical-distribution assumptions of the named version, or the replacement hypotheses are stated",
            "the required expectation and moment conditions are finite",
            "weak convergence in probability is distinguished from strong almost-sure convergence",
        ),
        "center the sample average at the correct expectation and state the correct convergence mode",
    ),
    "Stokes theorem": TheoremConditionContract(
        "Stokes theorem",
        (
            "the manifold or region is oriented and has the required smoothness",
            "the differential form has the required degree, regularity, and support",
            "the boundary carries the induced orientation",
        ),
        "match the integral of the exterior derivative with the correctly oriented boundary integral",
    ),
    "Green theorem": TheoremConditionContract(
        "Green theorem",
        (
            "the planar region and positively oriented piecewise-smooth boundary are specified",
            "the component functions have the required continuous partial derivatives on a neighborhood of the region",
        ),
        "preserve the sign convention relating circulation or flux to the corresponding double integral",
    ),
    "Newton iteration": TheoremConditionContract(
        "Newton iteration",
        (
            "the iteration function, derivative, and initial value are copied exactly from the statement",
            "the derivative denominator is nonzero at every reported iterate",
            "any convergence claim is separated from merely computing the requested iterates",
        ),
        "report the requested recurrence and iterates rather than replacing the required method by an exact root",
    ),
    "bisection": TheoremConditionContract(
        "bisection",
        (
            "the function is continuous on the stated interval",
            "the endpoint values have opposite signs or an endpoint is already a root",
            "the midpoint and retained half-interval follow the stated sign convention",
        ),
        "report the requested iterate or interval together with the correct width or error bound",
    ),
    "matrix-tree theorem": TheoremConditionContract(
        "matrix-tree theorem",
        (
            "the graph convention, labels, multiplicities, and directed or undirected setting are fixed",
            "the correct Laplacian is formed from the current graph",
            "the requested rooted or unrooted tree interpretation matches the chosen cofactor",
        ),
        "evaluate one correct Laplacian cofactor and preserve any root or orientation factor",
    ),
    "inclusion-exclusion": TheoremConditionContract(
        "inclusion-exclusion",
        (
            "the finite universe and excluded or included properties are defined without changing labels or repetition",
            "every intersection size used in the sum corresponds to the same domain",
        ),
        "include all intersection orders with alternating signs and audit the zero- and full-intersection cases",
    ),
    "maximum likelihood": TheoremConditionContract(
        "maximum likelihood",
        (
            "the sampling model, support, parameter space, and observed data are explicit",
            "parameter-dependent support and boundary parameter values are retained",
            "a stationary point is checked against the feasible boundaries before it is called a maximum",
        ),
        "state the estimator in the original parameterization and verify it maximizes the likelihood on the whole parameter space",
    ),
    "Smith normal form": TheoremConditionContract(
        "Smith normal form",
        (
            "the coefficient ring is a principal ideal domain for the stated form",
            "row and column operations are unimodular over that ring",
            "the diagonal invariant factors satisfy the divisibility chain",
        ),
        "reconstruct the required rank, quotient, or module invariants from the complete diagonal form",
    ),
}


def theorem_condition_protocol(spec, *, review: bool = False) -> str:
    """Render contracts only for theorem names explicit in the current input."""
    semantics = getattr(spec, "semantics", None)
    if semantics is None:
        return ""
    explicit = tuple(dict.fromkeys((
        *getattr(semantics, "named_theorems", ()),
        *getattr(semantics, "requested_methods", ()),
    )))
    contracts = [_CONTRACTS[name] for name in explicit if name in _CONTRACTS]
    return "\n".join(contract.render(review=review) for contract in contracts)
