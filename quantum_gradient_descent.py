from collections.abc import Callable
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unitaria.nodes.amplification.grover_amplification import GroverAmplification
from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.basic.scale import Scale
from unitaria.nodes.constants.constant_matrix import ConstantMatrix
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.node import Node


DEFAULT_CONSTANT_STEP = 0.5


def eval_poly(coeffs, x) -> float: # f(x) = sum_k C_k[x, ..., x], where coeffs[k] = C_k is a k-tensor
    x = np.asarray(x)
    total = np.asarray(coeffs[0], dtype=complex) if len(coeffs) else np.array(0.0)
    for k in range(1, len(coeffs)):
        term = np.asarray(coeffs[k], dtype=complex)
        for _ in range(k):
            term = np.tensordot(term, x, axes=([0], [0]))  # contract one axis with x
        total = total + term
    return float(np.real(total))


def grad_coeffs(coeffs) -> list: # A_j = (j+1) C_{j+1}; valid only for symmetric C_k
    return [(j + 1) * np.asarray(coeffs[j + 1]) for j in range(len(coeffs) - 1)]


def eval_grad(coeffs, x) -> np.ndarray:
    x = np.asarray(x)
    dim = x.shape[0]
    grad = np.zeros(dim, dtype=complex)
    for j, Gj in enumerate(grad_coeffs(coeffs)):
        term = np.asarray(Gj, dtype=complex)
        for _ in range(j):  # contract the last j axes with x, leaving the output axis
            term = np.tensordot(term, x, axes=([term.ndim - 1], [0]))
        grad = grad + term
    return grad


def _kron_power(a, p): # a \otimes ... \otimes a (p times)
    out = np.array(1.0)
    for _ in range(p):
        out = np.multiply.outer(out, a)
    return out


def example_problem_instance(degree=4) -> tuple[list, np.ndarray, np.ndarray]:
    Q = np.array([[1.5, 0.5], [0.5, 1.0]])
    x_star = np.array([1.0, -0.5])
    x_0 = np.array([0.5, 0.0])
    N = len(x_star)

    # f(x) = 1/2 (x - x*)^T Q (x - x*) + 1/degree sum_i (x_i - x*_i)^degree
    coeffs = [np.zeros((N,) * k) for k in range(degree + 1)]
    coeffs[0] = coeffs[0] + 0.5 * x_star @ Q @ x_star
    coeffs[1] = coeffs[1] - Q @ x_star
    coeffs[2] = coeffs[2] + 0.5 * Q
    if degree > 2:
        for i in range(N):
            for p in range(degree + 1):
                coeffs[p] = coeffs[p] + comb(degree, p) * (-x_star[i]) ** (degree - p) / degree * _kron_power(np.eye(N)[i], p)
    return coeffs, x_0, x_star


class ClassicalComputer:
    def armijo_step_size_classical(self, f: Callable[[np.ndarray], float], x: np.ndarray, grad: np.ndarray, d: np.ndarray, beta: float = 0.4, gamma: float = 0.25) -> float:
        alpha = 1.0
        # Backtrack until    '<='
        while f(x + alpha * d) > f(x) + gamma * alpha * np.real(grad @ d):
            alpha *= beta
        return alpha

    def constant_step_size_classical(self, eta: float = DEFAULT_CONSTANT_STEP) -> float:
        return float(eta)

    def barzilai_borwein_step_size_classical(self, s, y, long_step: bool = True, tol: float = 1e-16) -> float | None:
        sy = np.real(np.vdot(s, y))
        if sy <= tol:
            return None
        if long_step:
            ss = np.real(np.vdot(s, s))
            return float(ss / sy)
        yy = np.real(np.vdot(y, y))
        if yy <= tol:
            return None
        return float(sy / yy)

    def malitsky_mishchenko_step_size_classical(self, s, y, tol: float = 1e-16) -> float | None:
        y_norm = 2 * np.sqrt(np.real(np.vdot(y, y))) # Only the second term of the min in Malitsky2020 Alg. 1: |s| / (2|y|)
        if y_norm <= tol:
            return None
        s_norm = np.sqrt(np.real(np.vdot(s, s)))
        return float(s_norm / y_norm)

    def wolfe_powell_step_size_classical(self, f: Callable[[np.ndarray], float], df: Callable[[np.ndarray], np.ndarray], x: np.ndarray, d: np.ndarray, gamma: float = 0.4, rho: float = 0.6, tol: float = 1e-12, alpha_max: float = 1e10) -> float:
        alpha = 1.0
        phi = lambda a: f(x + a * d)
        g_phi = lambda a: np.real(np.vdot(df(x + a * d), d))
        psi = lambda a: phi(a) - f(x) - gamma * a * np.real(np.vdot(df(x), d))  # Armijo condition

        # Initialize interval
        if psi(alpha) <= 0.0: # Armijo condition
            if g_phi(alpha) >= rho * g_phi(0): # curvature condition
                return alpha
            a_ = alpha
            while psi(alpha) <= 0.0 and alpha < alpha_max:
                alpha *= 2.0
            b_ = alpha
        else:
            b_ = alpha
            while ((psi(alpha) > 0.0) or (g_phi(alpha) >= rho * g_phi(0.0))) and (alpha > tol):
                alpha /= 2.0
            a_ = alpha

        # Main loop
        alpha = 0.5 * (a_ + b_)
        while ((psi(alpha) > 0.0) or (g_phi(alpha) < rho * g_phi(0.0))) and (b_ - a_ > tol):
            if psi(alpha) > 0.0:
                b_ = alpha
            else:
                a_ = alpha
            alpha = 0.5 * (a_ + b_)
        return alpha

    def classical_gradient_descent(self, coeffs, x_0: np.ndarray, step_size: str, kmax: int = 10) -> np.ndarray:
        dim = x_0.shape[0]
        iterates = np.zeros((kmax + 1, dim), dtype=complex)
        iterates[0, :] = x_0.copy()

        f = lambda xx: eval_poly(coeffs, xx)
        df = lambda xx: eval_grad(coeffs, xx)

        # Remember the last gradient for the nonmonotone strategies Barzilai-Borwein and Malitsky-Mishchenko
        grad_prev = None

        for i in range(kmax):
            grad = df(iterates[i, :])
            d = -grad

            if "constant" in step_size:
                eta = self.constant_step_size_classical()
            elif "armijo" in step_size:
                eta = self.armijo_step_size_classical(f, iterates[i, :], grad, d)
            elif "wolfe" in step_size:
                eta = self.wolfe_powell_step_size_classical(f, df, iterates[i, :], d)
            elif "barzilai" in step_size:
                # This method can only start after the first step that has to be made with another strategy
                eta_bb = None
                if grad_prev is not None:
                    s = iterates[i, :] - iterates[i - 1, :]
                    # As in Dai-Fletcher (1.8) the first BB step at i = 1 is the short step BB2, so odd i takes BB2 and even i BB1.
                    eta_bb = self.barzilai_borwein_step_size_classical(s, grad - grad_prev, long_step=(i % 2 == 0))
                eta = eta_bb if eta_bb is not None else self.armijo_step_size_classical(f, iterates[i, :], grad, d)
            elif "malitsky" in step_size:
                eta_mm = None
                if grad_prev is not None:
                    s = iterates[i, :] - iterates[i - 1, :]
                    eta_mm = self.malitsky_mishchenko_step_size_classical(s, grad - grad_prev)
                eta = eta_mm if eta_mm is not None else self.armijo_step_size_classical(f, iterates[i, :], grad, d)
            else:
                raise ValueError(f"Unsupported step_size {step_size} for classical descent; use 'armijo', 'wolfe', 'barzilai', 'malitsky' or 'constant'.")

            grad_prev = grad
            iterates[i + 1, :] = iterates[i, :] + eta * d
        return iterates


# Keeps the amplified circuit, but returns the closed-form vector instead of simulating it
class AmplifiedVector(Node):
    def __init__(self, circuit_node: Node, vec: np.ndarray, normalization: float):
        super().__init__(1, vec.shape[0])
        self.circuit_node = circuit_node
        self.vec = vec
        self.amplified_normalization = normalization

    def children(self) -> list[Node]:
        return [self.circuit_node]

    def parameters(self) -> dict:
        return {"normalization": self.amplified_normalization}

    def _normalization(self) -> float:
        return self.amplified_normalization

    def _subspace_in(self):
        return self.circuit_node.subspace_in

    def _subspace_out(self):
        return self.circuit_node.subspace_out

    def compute(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            return self.vec * input[0]
        return (np.array([self.vec]).T @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.array([np.conj(self.vec)]) @ input.T).T

    def _circuit(self, target, clean_ancillae, borrowed_ancillae):
        return self.circuit_node.circuit(target, clean_ancillae, borrowed_ancillae)

    def clean_ancilla_count(self) -> int:
        return self.circuit_node.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.circuit_node.borrowed_ancilla_count()


class QuantumComputer:
    def __init__(self, seed=42, measure_tol=1e-4, amplify_tol=1e-5):
        self.gate_executions = 0
        self.rng = np.random.default_rng(seed)
        self.measure_tol = measure_tol  # absolute tol; readout accuracy
        self.amplify_tol = amplify_tol  # absolute tol; accuracy of the norm estimate before amplification

    def _create_blockencoding_poly(self, coeffs, x_0):
        self.dim = x_0.shape[0]
        self.x = ConstantVector(x_0)

        N = self.dim
        self.U_grad = []  # list of (j, Node)
        for j, Gj in enumerate(grad_coeffs(coeffs)):
            if not np.any(np.abs(Gj) > 1e-15):  # Skip zero blocks
                continue
            if j == 0:
                self.U_grad.append((0, ConstantVector(Gj.reshape(N))))
            else:
                self.U_grad.append((j, ConstantMatrix(Gj.reshape(N, N**j))))  # We flatten the tensor into a N x N^j block of a blockencoding

        self.U_obj = []  # list of (k, Node)
        for k in range(1, len(coeffs)):
            Ck = coeffs[k]
            if not np.any(np.abs(Ck) > 1e-15):
                continue
            self.U_obj.append((k, ConstantVector(Ck.reshape(N**k)).adjoint()))

    def _gate_count(self, node: Node) -> int:
        # Bottom up traversal that builds only the leaf circuits and counts each distinct subgraph once.
        counts = {}
        stack = [(node, False)]
        while stack:
            n, ready = stack.pop()
            nid = id(n)
            if nid in counts:
                continue
            # GroverAmplification hides its child. Its circuit applies A and A^dagger 2*iterations+1 times.
            if isinstance(n, GroverAmplification):
                children, multiplicity = [n.A], 2 * n.iterations + 1
            else:
                children, multiplicity = n.children(), 1

            if not children:
                counts[nid] = len(n._cached_circuit._tq_circuit.gates)
            elif ready:
                counts[nid] = multiplicity * sum(counts[id(c)] for c in children)
            else:
                stack.append((n, True))
                stack.extend((c, False) for c in children if id(c) not in counts)
        return counts[id(node)]

    def measure(self, x: Node, tol: float | None = None, n_gates: int | None = None) -> float:
        actual = x.toarray()
        value = float(np.real(np.asarray(actual).item()))
        tol = self.measure_tol if tol is None else tol
        n_gates = self._gate_count(x) if n_gates is None else n_gates
        self.gate_executions += np.ceil(x.normalization / tol) * n_gates  # shots * gates, where shots ~ normalization / tol.
        return value

    def measure_norm(self, x: Node, tol: float | None = None, n_gates: int | None = None) -> float:
        norm = float(np.linalg.norm(x.toarray()))
        tol = self.measure_tol if tol is None else tol
        n_gates = self._gate_count(x) if n_gates is None else n_gates
        self.gate_executions += np.ceil(x.normalization / tol) * n_gates
        return norm

    def amplify(self, x: Node, tol: float | None = None) -> Node:
        tol = self.amplify_tol if tol is None else tol
        norm = self.measure_norm(x, tol=tol) + self.rng.uniform(-tol, tol) # Uniform noise models the measurement error of the norm
        information_efficiency = norm / x.normalization
        if not 0.0 < information_efficiency < 1.0:
            return x
        # Smallest odd k with sin(pi/(2k)) <= a, so the amplified amplitude stays <= 1
        k = 2 * int(np.ceil(0.25 * (np.pi / np.arcsin(information_efficiency) - 2))) + 1
        remove_efficiency = information_efficiency / np.sin(np.pi / (2 * k))
        # remove_efficiency >= 1 mathematically, but it can round to exactly 1, which Scale rejects
        inner = x if remove_efficiency <= 1.0 else Scale(x, remove_efficiency=remove_efficiency)
        res = Scale(GroverAmplification(inner, (k - 1) // 2), norm, absolute=True)

        # Grover maps the amplitude a of inner to sin(k arcsin(a)) without turning the vector, so
        # the amplified value is this closed form and never has to be simulated through the QSVT.
        vec = x.toarray()
        amplitude = np.linalg.norm(vec) / inner.normalization
        vec = norm * np.sin(k * np.arcsin(min(amplitude, 1.0))) * vec / np.linalg.norm(vec)
        return AmplifiedVector(res, vec, norm)

    def _horner(self, blocks: dict, x_node: Node, lowest: int) -> Node | None:
        if not blocks:
            return None
        result = None
        for j in range(max(blocks), lowest - 1, -1):
            if result is not None:  # x & I_{N^j} maps x^{\otimes j} to x^{\otimes j+1}
                result = result @ (x_node if j == 0 else (x_node & Identity(dim=self.dim**j)))
            B = blocks.get(j)
            if B is not None:
                result = B if result is None else result + B
        return result

    def qf(self, x_node: Node) -> Node:
        result = self._horner(dict(self.U_obj), x_node, 1)  # C_1 + C_2 (x & I) + ..., a 1 x N row
        return None if result is None else result @ x_node

    def qdf(self, x_node: Node) -> Node:
        return self._horner(dict(self.U_grad), x_node, 0)

    def q_step(self, x_node: Node, eta: float) -> Node:
        blocks = dict(self.U_grad)
        # Horner down to level 1 folds the whole x-dependent part of grad f into one N x N
        # operator G with grad f(x) = A_0 + G x, so x_node is referenced once per degree.
        G = self._horner(blocks, x_node, 1)
        step = x_node if G is None else (Identity(dim=self.dim) - eta * G) @ x_node
        A_0 = blocks.get(0)
        return step if A_0 is None else step - eta * A_0

    def constant_step_size_quantum(self, eta: float = DEFAULT_CONSTANT_STEP) -> float:
        return float(eta)

    def armijo_step_size_quantum(self, x_node: Node, d_node: Node, grad_node: Node, beta: float = 0.4, gamma: float = 0.25) -> float:
        alpha = 1.0
        grad_congruence = self.measure(grad_node.adjoint() @ d_node)
        base = self.measure(self.qf(x_node))
        n_gates_phi = self._gate_count(self.qf(x_node + alpha * d_node))  # Can reuse the gate count here because it is independent of the scalar alpha.
        phi = lambda a: self.measure(self.qf(x_node + a * d_node), n_gates=n_gates_phi)

        # Backtracking loop (safeguarded with a minimum alpha)
        min_alpha = 1e-16
        while phi(alpha) > base + gamma * alpha * grad_congruence and alpha > min_alpha:
            alpha *= beta
        return alpha

    def wolfe_powell_step_size_quantum(self, f_node: Callable[[Node], Node], df_node: Callable[[Node], Node], x_node: Node, d_node: Node, gamma: float = 0.4, rho: float = 0.6, tol: float = 1e-12, alpha_max: float = 1e10) -> float:
        alpha = 1.0
        phi_node = lambda a: f_node(x_node + a * d_node)
        n_gates_phi = self._gate_count(phi_node(alpha))  # We can cache the gate count of phi and gphi because it is independent of alpha
        n_gates_gphi = self._gate_count(df_node(x_node + alpha * d_node).adjoint() @ d_node)
        # At a == 0 the node x + a d collapses, so that case is counted on its own.
        g_phi = lambda a: self.measure(df_node(x_node + a * d_node).adjoint() @ d_node, n_gates=n_gates_gphi if a != 0 else None)

        phi0_val = self.measure(phi_node(0.0))
        d_scalar_val = self.measure(df_node(x_node).adjoint() @ d_node)

        psi = lambda a: (self.measure(phi_node(a), n_gates=n_gates_phi) - phi0_val - gamma * a * d_scalar_val)

        # Initialize interval
        if psi(alpha) <= 0.0:
            if g_phi(alpha) >= rho * g_phi(0.0):
                return alpha
            a_ = alpha
            while psi(alpha) <= 0.0 and alpha < alpha_max:
                alpha *= 2.0
            b_ = alpha
        else:
            b_ = alpha
            while ((psi(alpha) > 0.0) or (g_phi(alpha) >= rho * g_phi(0.0))) and (alpha > tol):
                alpha /= 2.0
            a_ = alpha

        # Main loop
        alpha = 0.5 * (a_ + b_)
        while ((psi(alpha) > 0.0) or (g_phi(alpha) < rho * g_phi(0.0))) and (b_ - a_ > tol):
            if psi(alpha) > 0.0:
                b_ = alpha
            else:
                a_ = alpha
            alpha = 0.5 * (a_ + b_)
        return alpha

    def barzilai_borwein_step_size_quantum(self, s_node: Node, y_node: Node, long_step: bool = True, tol: float = 1e-16) -> float | None:
        sy = self.measure(s_node.adjoint() @ y_node)
        if sy <= tol:
            return None
        if long_step:
            ss = self.measure_norm(s_node) ** 2
            return float(ss / sy)
        yy = self.measure_norm(y_node) ** 2
        if yy <= tol:
            return None
        return float(sy / yy)

    def malitsky_mishchenko_step_size_quantum(self, s_node: Node, y_node: Node, tol: float = 1e-16) -> float | None:
        y_norm = 2 * self.measure_norm(y_node)
        if y_norm <= tol:
            return None
        s_norm = self.measure_norm(s_node)
        return float(s_norm / y_norm)

    def quantum_gradient_descent(self, coeffs, x_0: np.ndarray, step_size: str, kmax: int = 10, use_amplification: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dim = x_0.shape[0]

        # Prepare quality metrics
        iterates = np.zeros((kmax + 1, dim), dtype=complex)
        normalizations = np.zeros((kmax + 1))
        gate_execution_count = np.zeros((kmax + 1))
        gates_per_circuit = np.zeros((kmax + 1))

        self._create_blockencoding_poly(coeffs, x_0)

        iterates[0, :] = x_0.copy()
        normalizations[0] = self.x.normalization
        gate_execution_count[0] = self.gate_executions
        gates_per_circuit[0] = self._gate_count(self.x)

        # Remember the last gradient and iterate for the nonmonotone strategies Barzilai-Borwein and Malitsky-Mishchenko
        x_prev = None
        grad_prev = None

        for i in range(kmax):
            grad_node = self.qdf(self.x)
            d_node = (-1.0) * grad_node

            if "constant" in step_size:
                eta = self.constant_step_size_quantum()
            elif "armijo" in step_size:
                eta = self.armijo_step_size_quantum(self.x, d_node, grad_node)
            elif "wolfe" in step_size:
                eta = self.wolfe_powell_step_size_quantum(self.qf, self.qdf, self.x, d_node)
            elif "barzilai" in step_size:
                # This method can only start after the first step that has to be made with another strategy
                eta_bb = None
                if grad_prev is not None:
                    s_node = self.x - x_prev
                    y_node = grad_node - grad_prev
                    # As in Dai-Fletcher (1.8) the first BB step at i = 1 is the short step BB2, so odd i takes BB2 and even i BB1.
                    eta_bb = self.barzilai_borwein_step_size_quantum(s_node, y_node, long_step=(i % 2 == 0))
                eta = eta_bb if eta_bb is not None else self.armijo_step_size_quantum(self.x, d_node, grad_node)
            elif "malitsky" in step_size:
                eta_mm = None
                if grad_prev is not None:
                    s_node = self.x - x_prev
                    y_node = grad_node - grad_prev
                    eta_mm = self.malitsky_mishchenko_step_size_quantum(s_node, y_node)
                eta = eta_mm if eta_mm is not None else self.armijo_step_size_quantum(self.x, d_node, grad_node)
            else:
                raise ValueError(f"Unsupported step_size = {step_size!s} for quantum descent; use 'armijo', 'wolfe', 'barzilai', 'malitsky' or 'constant'.")

            x_old = self.x
            x_prev, grad_prev = x_old, grad_node
            self.x = self.q_step(x_old, eta)
            if use_amplification:
                self.x = self.amplify(self.x)

            iterates[i + 1, :] = np.real(self.x.toarray())
            normalizations[i + 1] = self.x.normalization
            gate_execution_count[i + 1] = self.gate_executions
            gates_per_circuit[i + 1] = self._gate_count(self.x)

            if not np.all(np.abs([normalizations[i + 1], gate_execution_count[i + 1], gates_per_circuit[i + 1]]) < 1e200): #Stop before the cost metrics overflow
                print(f"QuantumGradientDescent stopped after {i + 1} iterations using {step_size!s}: the cost metric got too large.")
                return iterates[: i + 1, :], normalizations[: i + 1], gate_execution_count[: i + 1], gates_per_circuit[: i + 1]

        return iterates, normalizations, gate_execution_count, gates_per_circuit


# colour -> step-size rule, marker -> classical ('o') vs quantum ('X'), line style -> classical (solid) vs quantum (dashed, dotted if amplified)
rule_color = {"armijo": "tab:orange", "wolfe": "tab:green", "barzilai": "tab:red", "malitsky": "tab:purple", "constant": "tab:brown"}
type_marker = {"classical": "o", "quantum": "X"}
type_line = {"classical": "-", "quantum": "--", "quantum_amplified": ":"}


def _style(r):
    rule = next((k for k in rule_color if k in r["name"]), "")
    line_key = "quantum_amplified" if r["type"] == "quantum" and r["amplified"] else r["type"]
    return dict(
        color=rule_color.get(rule, "tab:gray"),
        marker=type_marker[r["type"]],
        linestyle=type_line[line_key],
        markersize=8 if r["type"] == "quantum" else 6,
        markeredgecolor="k",
        markeredgewidth=0.5,
        alpha=0.85,
    )


def _save(fig, ax, name):
    ax.grid(True, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_dir = Path(__file__).resolve() / "result_graphics"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_landscape(coeffs, results, x_star):
    f_star = eval_poly(coeffs, x_star)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    paths = [np.real(r["iterates"]) for r in results]
    pts = np.vstack(paths + [x_star])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.15 * np.maximum(hi - lo, 1e-3)
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 150)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 150)
    Z = np.array([[eval_poly(coeffs, np.array([x, y])) for x in xs] for y in ys])
    # Log-spaced levels in the gap f - f* so the contour rings concentrate around the minimum.
    levels = f_star + np.geomspace((Z - f_star).min(), (Z - f_star).max(), 15) if np.all(Z - f_star > 0) else 15
    ax.contour(xs, ys, Z, levels=levels, cmap="viridis", linewidths=0.8, alpha=0.7)
    for r, it in zip(results, paths):
        ax.plot(it[:, 0], it[:, 1], label=r["name"], **_style(r))
    ax.scatter(x_star[0], x_star[1], marker="+", s=140, color="red", linewidths=1.6, zorder=7, label=r"$x^*$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    _save(fig, ax, "landscape")


def plot_iterate_error(results, x_star):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in results:
        err = np.linalg.norm(r["iterates"] - x_star, axis=1)
        ax.plot(range(len(err)), err, label=r["name"], **_style(r))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|x_k - x^*|$")
    ax.set_yscale("log")
    _save(fig, ax, "iterate_error")


def plot_error_vs_gate_executions(results, x_star):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in results:
        if r["gate_executions"] is None:
            continue
        err = np.linalg.norm(r["iterates"] - x_star, axis=1)
        ax.plot(r["gate_executions"][1:], err[1:], label=r["name"], **_style(r))
    ax.set_xlabel("Gate executions (cumulative)")
    ax.set_ylabel(r"$|x_k - x^*|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _save(fig, ax, "error_vs_gate_executions")


def plot_per_iteration(results, key, ylabel, name):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in results:
        if r[key] is None:
            continue
        ax.plot(range(len(r[key])), r[key], label=r["name"], **_style(r))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    _save(fig, ax, name)


def plot_readout_cost(results, measure_tol):
    # Gate executions so far plus one measurement of the norm of the current iterate.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in results:
        readout = r["gate_executions"] + r["normalizations"] * r["gates_per_circuit"] / measure_tol
        color = "tab:blue" if r["degree"] == 2 else "tab:red"
        linestyle = type_line["quantum_amplified" if r["amplified"] else "quantum"]
        ax.plot(range(len(readout)), readout, color=color, linestyle=linestyle, marker="X", markeredgecolor="k", markeredgewidth=0.5, label=f"degree {r['degree']}, {r['name']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gate executions to read out the iterate")
    ax.set_yscale("log")
    _save(fig, ax, "readout_cost")


def _run_job(job: dict) -> dict:
    kind, s = job["kind"], job["step"]
    coeffs, x_0, kmax, amplification = job["coeffs"], job["x_0"], job["kmax"], job["amplification"]

    if kind == "classical":
        iterates = ClassicalComputer().classical_gradient_descent(coeffs, x_0, s, kmax=kmax)
        return {
            "name": f"classical_{s}",
            "type": "classical",
            "amplified": False,
            "iterates": iterates,
            "gate_executions": None,
            "normalizations": None,
            "gates_per_circuit": None,
        }

    q_iters, q_norms, q_gate_execs, q_gates_pc = QuantumComputer().quantum_gradient_descent(coeffs, x_0, s, kmax=kmax, use_amplification=amplification)
    return {
        "name": f"quantum_{s}" + ("+amp" if amplification else ""),
        "type": "quantum",
        "amplified": amplification,
        "iterates": q_iters,
        "gate_executions": q_gate_execs,
        "normalizations": q_norms,
        "gates_per_circuit": q_gates_pc,
    }


def main():
    degree = 4  # Must be even, else f is unbounded below.
    coeffs, x_0, x_star = example_problem_instance(degree)

    kmax = 6
    amplification = False

    step_names = ["armijo", "wolfe", "barzilai", "malitsky", "constant"]

    jobs = [
        {
            "kind": kind,
            "step": s,
            "coeffs": coeffs,
            "x_0": x_0,
            "kmax": kmax,
            "amplification": amplification,
        }
        for s in step_names
        for kind in ("classical", "quantum")
    ]
    results = [_run_job(job) for job in jobs]

    plot_landscape(coeffs, results, x_star)
    plot_iterate_error(results, x_star)
    plot_error_vs_gate_executions(results, x_star)
    plot_per_iteration(results, "normalizations", "Normalization factor", "normalization")
    plot_per_iteration(results, "gates_per_circuit", "Gates per circuit", "gates_per_circuit")
    plot_per_iteration(results, "gate_executions", "Gate executions (cumulative)", "gate_executions")

    # Constant step size with and without amplification for degree 2 and degree 4
    readout_results = []
    for degree in (2, 4):
        coeffs, x_0, x_star = example_problem_instance(degree)
        for amplification in (True, False):
            r = _run_job({"kind": "quantum", "step": "constant", "coeffs": coeffs, "x_0": x_0, "kmax": kmax, "amplification": amplification})
            r["degree"] = degree
            readout_results.append(r)
    plot_readout_cost(readout_results, QuantumComputer().measure_tol)


if __name__ == "__main__":
    main()
