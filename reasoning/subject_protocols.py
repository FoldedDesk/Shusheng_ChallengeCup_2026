"""Short subject-specific solve and audit protocols for unseen problems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectProtocol:
    identifier: str
    solve_steps: tuple[str, ...]
    audit_steps: tuple[str, ...]

    def render(self, *, review: bool, language: str) -> str:
        steps = self.audit_steps if review else self.solve_steps
        heading = (
            "学科复核协议" if review else "学科求解协议"
        ) if language == "zh" else (
            "Subject audit protocol" if review else "Subject solve protocol"
        )
        return heading + ": " + " ".join(
            f"{index}. {step}" for index, step in enumerate(steps, start=1)
        )


_PROTOCOLS = {
    "离散数学": SubjectProtocol(
        "discrete",
        (
            "Define the counted or graph object, labels, order, repetition, and symmetry exactly.",
            "Choose a structural certificate: bijection, recurrence, invariant, inclusion-exclusion, generating function, or matrix-tree computation.",
            "Prove exhaustiveness and evaluate the smallest legal cases before stating the result.",
        ),
        (
            "Recount by a different method or enumerate a small instance.",
            "Check integrality, index shifts, symmetry factors, connectivity, and degenerate objects.",
            "Try to construct an omitted or multiply counted object.",
        ),
    ),
    "数值分析": SubjectProtocol(
        "numerical",
        (
            "Write the requested algorithm, data, step size, stopping rule, and error sign convention before computing.",
            "Derive the approximation without replacing the requested method by an exact solve.",
            "Report the requested precision together with a residual or error bound.",
        ),
        (
            "Recompute the residual in the original equation or scheme.",
            "Check consistency order, convergence hypotheses, stability region, endpoints, and rounding separately.",
            "Verify every requested node, weight, iterate, and error sign.",
        ),
    ),
    "测度积分": SubjectProtocol(
        "measure",
        (
            "State the measure space, measurability, convergence mode, and exceptional sets.",
            "Name the exact convergence or integration theorem and verify each hypothesis before interchange.",
            "Separate almost-everywhere statements from integral or norm conclusions.",
        ),
        (
            "Search for moving-spike, infinite-measure, nonmeasurable, or nonintegrable counterexamples.",
            "Check whether one common integrable dominator or absolute integrability was actually established.",
            "Audit every quantifier and null-set dependency.",
        ),
    ),
    "微分几何": SubjectProtocol(
        "differential_geometry",
        (
            "Fix the manifold, chart, metric, orientation, and regularity assumptions.",
            "Compute invariant quantities from one consistent convention for frames, normals, forms, or connections.",
            "State where the formula is valid and simplify only after substitution.",
        ),
        (
            "Check regularity determinants, tensor dimensions, and orientation dependence.",
            "Recompute through an invariant identity, special point, or alternate coordinates.",
            "Verify signs in curvature, boundary orientation, and Gauss-Bonnet terms.",
        ),
    ),
    "概率论": SubjectProtocol(
        "probability",
        (
            "Define the sample space, support, conditioning information, and independence assumptions.",
            "Use conditioning, indicators, transforms, or symmetry with all normalizing constants visible.",
            "Check total mass before computing probabilities or expectations.",
        ),
        (
            "Verify probabilities are normalized and test complement and degenerate parameter cases.",
            "Recompute by a different conditioning direction or first-step equation.",
            "Check expectation bounds and never infer independence from disjointness.",
        ),
    ),
    "随机过程": SubjectProtocol(
        "stochastic_process",
        (
            "Specify states, time direction, filtration, transition law, and boundary behavior.",
            "Derive first-step, generator, martingale, or independent-increment equations with initial data.",
            "Solve the equations and enforce normalization and boundary conditions.",
        ),
        (
            "Substitute into transition, generator, or boundary equations.",
            "Check stopping-time hypotheses, recurrence classes, and limiting versus stationary claims.",
            "Test absorbing and deterministic edge cases.",
        ),
    ),
    "抽象代数": SubjectProtocol(
        "abstract_algebra",
        (
            "Name the group, ring, field, module, base object, morphisms, and finiteness assumptions.",
            "Derive restrictions from kernels, images, orders, ideals, degrees, or canonical forms.",
            "Prove both existence and exclusion when classifying an isomorphism type.",
        ),
        (
            "Compare order, dimension, rank, exponent, torsion, and quotient invariants.",
            "Check polynomial irreducibility by exact roots, gcds, or factorization over the stated base field; recompose every claimed factorization.",
            "Attempt a small counterexample over a different base ring or field.",
        ),
    ),
    "复分析": SubjectProtocol(
        "complex_analysis",
        (
            "Locate singularities and zeros, specify the domain and contour orientation, and classify each local behavior.",
            "Apply Cauchy, residues, argument principle, or local series only within its hypotheses.",
            "Include every enclosed contribution and the orientation factor.",
        ),
        (
            "Recompute residues from a local expansion or derivative formula.",
            "Check poles on the contour, winding numbers, branch choices, and behavior at infinity.",
            "Compare with symmetry or direct parameterization when available.",
        ),
    ),
    "常微分方程": SubjectProtocol(
        "ode",
        (
            "Identify equation class, solve the homogeneous part, construct a particular solution, and then impose all data.",
            "Track resonance, maximal intervals, and constants before simplification.",
            "Present one explicit solution satisfying every initial or boundary condition.",
        ),
        (
            "Differentiate and substitute into the original differential equation.",
            "Evaluate every initial or boundary condition exactly.",
            "Check uniqueness hypotheses and singular points; for a one-sided interval request, state both the full maximal interval containing the initial point and the requested side.",
            "For equilibrium classification, state the eigenvalues, their signs, the phase type, and explicitly whether it is stable or unstable.",
        ),
    ),
    "偏微分方程": SubjectProtocol(
        "pde",
        (
            "State the domain, PDE type, initial-boundary data, and intended classical, weak, or distributional solution class.",
            "Choose characteristics, separation, transform, energy, maximum principle, or weak formulation with hypotheses checked.",
            "Normalize fundamental solutions and enforce all boundary data.",
        ),
        (
            "Apply the differential operator or test the weak identity directly.",
            "Check boundary and initial traces, constants, signs, and singular normalization.",
            "Use uniqueness or an energy identity to rule out alternatives.",
        ),
    ),
    "泛函分析": SubjectProtocol(
        "functional_analysis",
        (
            "Specify the metric or normed spaces, completeness assumptions, operator domains, scalar field, and topology before invoking a theorem.",
            "For metric-space questions work directly with Cauchy sequences and limits; for operator questions separate boundedness, compactness, spectrum, and norm attainment.",
            "Prove the exact requested implication, and for a sharp norm or constant give both an upper bound and an attaining vector or extremizing sequence.",
        ),
        (
            "Check metric, domains, density, completeness, and inner-product conventions before using a completeness or operator theorem.",
            "Substitute a proposed eigenfunction or extremizer and recompute its norm ratio.",
            "Test standard noncompact or nonreflexive counterexamples when a converse is claimed.",
        ),
    ),
    "统计推断": SubjectProtocol(
        "statistical_inference",
        (
            "State the sampling model, support, parameter space, statistic, and target estimand before simplifying the likelihood.",
            "Prove sufficiency, completeness, unbiasedness, consistency, or optimality using the theorem actually requested.",
            "Keep parameter-dependent support and all normalizing constants explicit.",
        ),
        (
            "Normalize the density and recompute the estimator expectation and variance under the original model.",
            "Audit regularity, completeness, identifiability, and parameter-space hypotheses rather than naming a theorem alone.",
            "Check boundary parameter values and sufficient-statistic reductions directly.",
        ),
    ),
    "线性回归": SubjectProtocol(
        "linear_regression",
        (
            "Write the design matrix, response vector, covariance assumptions, parameter dimensions, and rank conditions.",
            "Derive OLS or GLS from the appropriate normal equations and distinguish estimates from fitted values and residuals.",
            "State any unbiasedness, covariance, projection, or distributional conclusion with its assumptions.",
        ),
        (
            "Check matrix dimensions, invertibility or generalized-inverse conditions, and the normal-equation residual.",
            "Recompute bias and covariance and verify weighted orthogonality for GLS.",
            "Test whether an intercept, heteroskedasticity, or correlation changes the claimed formula.",
        ),
    ),
    "非基础及进阶课程": SubjectProtocol(
        "advanced",
        (
            "Identify the precise mathematical objects and subfield from the statement before selecting a theorem.",
            "Derive the requested invariant, extremum, or construction from definitions with normalization and hypotheses visible.",
            "Include an equality case, boundary condition, or reconstruction check whenever the result is sharp.",
        ),
        (
            "Check dimensions, signs, scaling, normalization, and boundary terms independently.",
            "Recompute one decisive identity from the original definitions.",
            "Try degenerate and lowest-dimensional cases to expose an overgeneralized theorem application.",
        ),
    ),
    "高等代数": SubjectProtocol(
        "advanced_linear_algebra",
        (
            "Translate the problem into subspaces, ranks, invariant factors, minimal polynomials, or canonical blocks.",
            "Determine every multiplicity from dimension and rank data before naming the canonical form.",
            "Reconstruct the original invariants from the proposed decomposition.",
        ),
        (
            "Check total dimension, ranks of powers, trace, determinant, and minimal-polynomial constraints.",
            "Verify block multiplicities are nonnegative integers and exhaust the space.",
            "Rebuild all supplied invariants from the final normal form.",
        ),
    ),
    "运筹学": SubjectProtocol(
        "operations_research",
        (
            "Define decision variables, objective sense, feasible constraints, integrality, and boundedness.",
            "Use vertices, simplex conditions, dynamic programming, network structure, or duality as appropriate.",
            "Give a feasible optimizer and a complete optimality certificate.",
        ),
        (
            "Substitute the proposed solution into every constraint and recompute its objective value.",
            "Enumerate all relevant active sets or verify a dual feasible solution and complementary slackness.",
            "Check infeasibility, unbounded directions, degeneracy, and integer restrictions separately.",
        ),
    ),
    "数学分析": SubjectProtocol(
        "analysis",
        (
            "State domains, quantifiers, convergence mode, endpoint behavior, and regularity before taking limits or derivatives.",
            "Use epsilon estimates, compactness, monotonicity, convexity, or a convergence theorem with every hypothesis checked.",
            "Distinguish pointwise, uniform, integral, and norm conclusions.",
        ),
        (
            "Test endpoints, moving peaks, nonuniformity, and interchange of limits with integrals or derivatives.",
            "Produce an explicit bound independent of the limiting index when uniform control is claimed.",
            "Check equality cases, signs, and sharp constants directly.",
        ),
    ),
    "拓扑学": SubjectProtocol(
        "topology",
        (
            "Specify the spaces, topology, basepoints, maps, quotient or attachment data, and the invariant requested.",
            "Apply definitions, van Kampen, exact sequences, homology, or compactness theorems with intersection hypotheses explicit.",
            "Reduce presentations or invariants completely before stating the isomorphism type.",
        ),
        (
            "Check basepoint and path-connectivity assumptions, attachment relations, and induced maps.",
            "Compare fundamental group, homology, connectedness, compactness, and separation invariants.",
            "Search standard quotient and separation counterexamples for any claimed converse.",
        ),
    ),
}


_EXTRA_PROTOCOLS = {
    "数论": SubjectProtocol(
        "number_theory",
        (
            "Translate divisibility, congruence, valuation, and coprimality conditions exactly before manipulating them.",
            "Use one structural route such as modular arithmetic, descent, factorization, valuations, or orders, with every exceptional prime separated.",
            "Prove exhaustiveness and substitute every proposed integer family into the original condition.",
        ),
        (
            "Check small and boundary integers without treating them as a proof of the general case.",
            "Recompute residues and valuations at every exceptional prime and verify all divisibility directions.",
            "Test whether cancellation, division, or an order argument silently assumes coprimality.",
        ),
    ),
    "欧氏几何": SubjectProtocol(
        "euclidean_geometry",
        (
            "Record incidence, order, cyclicity, tangency, and directed-angle conventions before adding auxiliary points.",
            "Choose one coherent route: angle chase, similarity, power, coordinates, vectors, inversion, or projective structure.",
            "Derive the requested relation and check degeneracy and orientation explicitly.",
        ),
        (
            "Verify every claimed collinearity, cyclic quadrilateral, ratio, and angle orientation from the stated configuration.",
            "Recompute the decisive identity in coordinates or through a second geometric invariant.",
            "Check limiting, symmetric, and near-degenerate configurations for hidden assumptions.",
        ),
    ),
    "进阶数学": SubjectProtocol(
        "general_advanced",
        (
            "Identify the exact objects, quantifiers, and requested conclusion before selecting a field-specific theorem.",
            "Commit to one derivation from definitions and make every theorem hypothesis and exceptional case explicit.",
            "End with the requested object or conclusion and one concrete falsification check.",
        ),
        (
            "Recompute one decisive equality, invariant, boundary case, or counterexample from the original statement.",
            "Check domains, quantifiers, equality cases, normalization, and all excluded cases.",
            "Reject an argument that only cites authority or repeats the proposed conclusion.",
        ),
    ),
}


def subject_protocol(
    spec,
    *,
    review: bool = False,
    subject_override: str = "",
) -> str:
    subject = subject_override or getattr(
        spec.profile, "primary_subject", spec.profile.subject
    )
    protocol = _PROTOCOLS.get(subject) or _EXTRA_PROTOCOLS.get(subject)
    if protocol is None:
        topic = str(getattr(spec.profile, "topic", ""))
        topic_subject = {
            "olympiad_combinatorics": "离散数学",
            "olympiad_sequence": "离散数学",
            "combinatorics": "离散数学",
            "graph": "离散数学",
            "olympiad_number_theory": "数论",
            "olympiad_geometry": "欧氏几何",
            "olympiad_functional_equation": "高等代数",
            "olympiad_inequality": "高等代数",
            "olympiad_polynomial": "高等代数",
        }.get(topic, "")
        protocol = _PROTOCOLS.get(topic_subject) or _EXTRA_PROTOCOLS.get(topic_subject)
    if protocol is None:
        return ""
    return protocol.render(review=review, language=spec.profile.language)
