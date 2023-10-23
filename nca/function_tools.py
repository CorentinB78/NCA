import numpy as np
from numpy import fft
from scipy import integrate, interpolate
from .utilities import print_warning_large_error


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** (len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def next_odd_regular(target):
    """
    Find the next odd reglar number greater than or equal to target.
    THese are composites of prime factors 3 and 5 only.
    """
    N = _next_regular(target)
    while (N % 2) == 0:
        N = _next_regular(N + 1)

    return N


class Mesh:
    def __init__(self, xmax, nr_samples, pt_on_value=False, adjust_nr_samples=True):
        if adjust_nr_samples:
            nr_samples = next_odd_regular(nr_samples)

        if nr_samples % 2 != 1:
            raise ValueError("nr of samples must be odd.")

        if pt_on_value > 0.0:
            delta = 2 * xmax / (nr_samples - 1)
            x_next = delta * (pt_on_value // delta + 1)
            xmax *= pt_on_value / x_next
            if pt_on_value > xmax:
                raise ValueError("`pt_on_value` cannot be reached. Increase xmax.")

        self.xmin = -xmax
        self.xmax = xmax
        self.nr_samples = nr_samples
        self.delta = (xmax - self.xmin) / (nr_samples - 1)
        self.data = None

        self.pt_on_value = pt_on_value
        # TODO: test pt_on_value
        self.idx_pt_on_value = int((pt_on_value - self.xmin) / self.delta)
        self.pt_on_value_adj = False

    def values(self):
        if self.data is None:
            self.data = np.linspace(self.xmin, self.xmax, self.nr_samples)
            self.data[self.nr_samples // 2] = 0.0  # enforce exact zero
        return self.data

    def adjoint(self):
        out = Mesh(
            2 * np.pi * (self.nr_samples - 1) / (2 * self.delta * self.nr_samples),
            self.nr_samples,
            adjust_nr_samples=False,
        )

        out.pt_on_value = self.pt_on_value
        out.idx_pt_on_value = self.idx_pt_on_value
        out.pt_on_value_adj = not self.pt_on_value_adj

        return out

    def __len__(self):
        return self.nr_samples

    def __eq__(self, other):
        return (self.delta == other.delta) and (self.nr_samples == other.nr_samples)

    def __array__(self, dtype=None):
        return np.asarray(self.values(), dtype=dtype)


def interp(mesh_a, mesh_b, func_b, kind="linear", allow=True):
    if mesh_a is mesh_b:
        return func_b
    if not allow:
        raise RuntimeError
    return interpolate.interp1d(
        mesh_b.values(),
        func_b,
        kind=kind,
        assume_sorted=True,
        copy=False,
        bounds_error=False,
        fill_value=0.0,
    )(mesh_a.values())


def checked_interp(mesh_a, mesh_b, func_b, kind="cubic", tol=1e-3):
    if mesh_a is mesh_b:
        return func_b
    mesh_a_half = Mesh(
        mesh_a.xmax, 2 * (mesh_a.nr_samples // 4) + 1, adjust_nr_samples=False
    )

    vals1 = interp(mesh_a, mesh_b, func_b, kind=kind, allow=True)
    vals2 = interp(mesh_a_half, mesh_b, func_b, kind=kind, allow=True)
    check = interp(mesh_a, mesh_a_half, vals2, kind="linear", allow=True)

    err = np.sum(np.abs(check - vals1)) * mesh_a.delta

    print_warning_large_error(
        f"Low number of samples for this interpolation. err={err}",
        err,
        tolw=tol,
        tole=1e-1,
    )

    return vals1


def product_functions(mesh_a, func_a, mesh_b, func_b):
    """
    Interpolate on the smaller mesh
    """
    if mesh_a.xmax > mesh_b.xmax:
        return product_functions(mesh_b, func_b, mesh_a, func_a)

    func_b = interp(mesh_a, mesh_b, func_b)
    return mesh_a, func_a * func_b


def sum_functions(mesh_a, func_a, mesh_b, func_b):
    """
    Interpolate on the larger mesh. Is this the good choice?
    """
    if mesh_a is None:
        return mesh_b, func_b

    if mesh_a.xmax < mesh_b.xmax:
        return sum_functions(mesh_b, func_b, mesh_a, func_a)

    func_b = interp(mesh_a, mesh_b, func_b)
    return mesh_a, func_a + func_b


def fourier_transform(mesh, f, axis=-1):
    adj_mesh = mesh.adjoint()
    f = np.swapaxes(f, -1, axis)
    g = fft.fftshift(fft.fft(f, axis=-1), axes=-1)[..., ::-1]
    g *= mesh.delta * np.exp(adj_mesh.values() * (1j * mesh.xmin))
    return adj_mesh, np.swapaxes(g, -1, axis)


def inv_fourier_transform(mesh, f, axis=-1):
    adj_mesh = mesh.adjoint()
    f = np.swapaxes(f, -1, axis)
    g = fft.fftshift(fft.fft(f, axis=-1), axes=-1)
    g *= (mesh.delta / (2 * np.pi)) * np.exp(adj_mesh.values() * (-1j * mesh.xmin))
    return adj_mesh, np.swapaxes(g, -1, axis)


def planck_taper_window(mesh, W, eps):
    Wp = W + eps / 2.0
    Wm = W - eps / 2.0
    assert Wm > 0.0
    out = np.empty(len(mesh))
    for k, x in enumerate(mesh.values()):
        if np.abs(x) >= Wp:
            out[k] = 0
        elif np.abs(x) > Wm:
            out[k] = 0.5 * (
                1.0
                - np.tanh((Wp - Wm) / (Wp - np.abs(x)) - (Wp - Wm) / (np.abs(x) - Wm))
            )
        else:
            out[k] = 1.0
    return out

#### Alpert Fourier transform ####

def _get_alpert_regular_rule(order: int):
    """
    Returns the elements of the Alpert quadrature rule for non-singular functions.
    
    Available orders are 0, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32.

    Returns a, x, w
    """
    if order == 0:
        a = 0

        x = []

        w = []

    elif order == 3:
        a = 1

        x = [1.666666666666667e-01]

        w = [5.000000000000000e-01]

    elif order == 4:
        a = 2

        x = [2.000000000000000e-01,
            1.000000000000000e+00]

        w = [5.208333333333333e-01,
            9.791666666666667e-01]

    elif order == 5:
        a = 2

        x = [2.245784979812614e-01,
            1.013719374359164e+00]

        w = [5.540781643606372e-01,
            9.459218356393628e-01]

    elif order == 6:
        a  = 3

        x = [2.250991042610971e-01,
            1.014269060987992e+00,
            2.000000000000000e+00]

        w = [5.549724327164180e-01,
            9.451317411845473e-01,
            9.998958260990347e-01]

    elif order == 7:
        a = 3

        x = [2.180540672543505e-01,
            1.001181873031216e+00,
            1.997580526418033e+00]

        w = [5.408088967208193e-01,
            9.516615045823566e-01,
            1.007529598696824e+00]

    elif order == 8:
        a = 4

        x = [2.087647422032129e-01,
            9.786087373714483e-01,
            1.989541386579751e+00,
            3.000000000000000e+00]

        w = [5.207988277246498e-01,
            9.535038018555888e-01,
            1.024871626402471e+00,
            1.000825744017291e+00]

    elif order == 12:
        a = 5

        x = [7.023955461621939e-02,
            4.312297857227970e-01,
            1.117752734518115e+00,
            2.017343724572518e+00,
            3.000837842847590e+00,
            4.000000000000000e+00]

        w = [1.922315977843698e-01,
            5.348399530514687e-01,
            8.170209442488760e-01,
            9.592111521445966e-01,
            9.967143408044999e-01,
            9.999820119661890e-01]

    elif order == 16:
        a = 7

        x = [9.919337841451028e-02,
            5.076592669645529e-01,
            1.184972925827278e+00,
            2.047493467134072e+00,
            3.007168911869310e+00,
            4.000474996776184e+00,
            5.000007879022339e+00,
            6.000000000000000e+00]

        w = [2.528198928766921e-01,
            5.550158230159486e-01,
            7.852321453615224e-01,
            9.245915673876714e-01,
            9.839350200445296e-01,
            9.984463448413151e-01,
            9.999592378464547e-01,
            9.999999686258662e-01]

    elif order == 20:
        a = 9

        x = [9.209200446233291e-02,
            4.752021947758861e-01,
            1.124687945844539e+00,
            1.977387385642367e+00,
            2.953848957822108e+00,
            3.976136786048776e+00,
            4.994354281979877e+00,
            5.999469539335291e+00,
            6.999986704874333e+00,
            8.000000000000000e+00]

        w = [2.351836144643984e-01,
            5.248820509085946e-01,
            7.634026409869887e-01,
            9.284711336658351e-01,
            1.010969886587741e+00,
            1.024959725311073e+00,
            1.010517534639652e+00,
            1.001551595797932e+00,
            1.000061681794188e+00,
            1.000000135843597e+00]

    elif order == 24:
        a = 10

        x = [6.001064731474805e-02,
            3.149685016229433e-01,
            7.664508240518316e-01,
            1.396685781342510e+00,
            2.175195903206602e+00,
            3.062320575880355e+00,
            4.016440988792476e+00,
            5.002872064275734e+00,
            6.000285453310164e+00,
            7.000012964962529e+00,
            8.000000175554469e+00,
            9.000000000000000e+00]

        w = [1.538932104518340e-01,
            3.551058128559424e-01,
            5.449200036280007e-01,
            7.104078497715549e-01,
            8.398780940253654e-01,
            9.272767950890611e-01,
            9.750605697371132e-01,
            9.942629650823470e-01,
            9.992421778421898e-01,
            9.999534370786161e-01,
            9.999990854912925e-01,
            9.999999989466828e-01]

    elif order == 28:
        a = 12

        x = [6.234360533194102e-02,
            3.250286721702614e-01,
            7.837350794282182e-01,
            1.415673112616924e+00,
            2.189894250061313e+00,
            3.070053877483040e+00,
            4.018613756218047e+00,
            5.002705902035397e+00,
            5.999929741810400e+00,
            6.999904720846024e+00,
            7.999986894843540e+00,
            8.999999373380393e+00,
            9.999999992002911e+00,
            1.100000000000000e+01]

        w = [1.595975279734157e-01,
            3.637046028193864e-01,
            5.498753177297441e-01,
            7.087986792086956e-01,
            8.335172275501195e-01,
            9.204446510608518e-01,
            9.710881776552090e-01,
            9.933296578555239e-01,
            9.994759087910050e-01,
            1.000133030254421e+00,
            1.000032915011460e+00,
            1.000002261653775e+00,
            1.000000042393520e+00,
            1.000000000042872e+00]

    elif order == 32:
        a = 14

        x = [5.899550614325259e-02,
            3.082757062227814e-01,
            7.463707253079130e-01,
            1.355993726494664e+00,
            2.112943217346336e+00,
            2.987241496545946e+00,
            3.944798920961176e+00,
            4.950269202842798e+00,
            5.972123043117706e+00,
            6.989783558137742e+00,
            7.997673019512965e+00,
            8.999694932747039e+00,
            9.999979225211805e+00,
            1.099999938266130e+01,
            1.199999999462073e+01,
            1.300000000000000e+01]

        w = [1.511076023874179e-01,
            3.459395921169090e-01,
            5.273502805146873e-01,
            6.878444094543021e-01,
            8.210319140034114e-01,
            9.218382875515803e-01,
            9.873027487553060e-01,
            1.018251913441155e+00,
            1.021933430349293e+00,
            1.012567983413513e+00,
            1.004052289554521e+00,
            1.000713413344501e+00,
            1.000063618302950e+00,
            1.000002486385216e+00,
            1.000000030404477e+00,
            1.000000000020760e+00]

    else:
        raise ValueError(f"There is no Alpert rule for order {order}. Available orders are 0, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32.")

    return a, np.array(x), np.array(w)


class AlpertMeshFunction:

    def __init__(self, delta_t, M, order):
        """
        Initialize function stored on a mesh adapted for the Alpert rule.
        
        Parameters:
         * delta_t (float): step in the central section
         * M (int): number of values in the central section
         * order (int): order of the Alpert rule. If zero, the mesh is simply a regular grid from 0 to tmax, and the right and left sections are empty.
        """
        a, x, w = _get_alpert_regular_rule(order)
        self.tmax = (M + 2 * a - 1) * delta_t
        self.order = order
        self.delta_t = delta_t
        self.M = M
        self.a = a
        self.alpert_weights = w
        self.times_left = delta_t * x
        if order > 0:
            assert(self.times_left[-1] < self.tmax)
        self.times_center = delta_t * (a + np.arange(M))
        self.times_right = self.tmax - self.times_left

        self.values_left = None
        self.values_center = None
        self.values_right = None

    def has_same_rule_as(self, other):
        return self.order == other.order and self.M == other.M and self.delta_t == other.delta_t

    def conj(self):
        out = self.get_empty_duplicate()
        out.values_left = np.conj(self.values_left)
        out.values_center = np.conj(self.values_center)
        out.values_right = np.conj(self.values_right)
        return out

    def __iadd__(self, other):
        if not(self.has_same_rule_as(other)):
            raise ValueError("Cannot add AlpertMeshFunctions with different Alpert rules")

        if self.values_left is None: # object was not populated and represents the zero function
            self.values_left = other.values_left
            self.values_center = other.values_center
            self.values_right = other.values_right
        else:
            self.values_left += other.values_left
            self.values_center += other.values_center
            self.values_right += other.values_right
        return self

    def __mul__(self, other):
        new = self.get_empty_duplicate()
        if isinstance(other, AlpertMeshFunction):
            if not(self.has_same_rule_as(other)):
                raise ValueError("Cannot multiply AlpertMeshFunctions with different Alpert rules")

            new.values_left = self.values_left * other.values_left
            new.values_center = self.values_center * other.values_center
            new.values_right = self.values_right * other.values_right
        else:
            raise NotImplementedError
        return new

    def __imul__(self, other):
        """ in place multiplication with scalar"""
        if isinstance(other, AlpertMeshFunction):
            if not(self.has_same_rule_as(other)):
                raise ValueError("Cannot multiply AlpertMeshFunctions with different Alpert rules")
            self.values_left *= other.values_left
            self.values_center *= other.values_center
            self.values_right *= other.values_right
        elif isinstance(other, complex):
            self.values_left *= other
            self.values_center *= other
            self.values_right *= other
        else:
            raise NotImplementedError
        return self

    def get_empty_duplicate(self):
        return AlpertMeshFunction(self.delta_t, self.M, self.order)

    def isfinite(self):
        return np.isfinite(self.values_left).all() and np.isfinite(self.values_right).all() and np.isfinite(self.values_center).all()

    def check_integrity(self):
        if self.values_left is None:
            assert self.values_center is None and self.values_right is None
        else:
            assert len(self.times_left) == len(self.values_left)
            assert len(self.times_center) == len(self.values_center)
            assert len(self.times_right) == len(self.values_right)


def make_alpert(delta_t, M, order, f):
    out = AlpertMeshFunction(delta_t, M, order)
    out.values_left = f(out.times_left)
    out.values_right = f(out.times_right)
    out.values_center = f(out.times_center)

    if not out.isfinite():
        raise ValueError("Evaluation of `f` gave non finite values")

    return out


def alpert_fourier_transform(alpert_function, wmin, N):
    """
    Fourier transform of function using the Alpert rule.
    
    Returns values at frequencies wmin + 2 pi dt k / N, for k=0..N-1 
    """
    f = alpert_function
    if N < f.M:
        raise ValueError
    dw = 2 * np.pi / (N * f.delta_t)
    w_samples = wmin + np.arange(N) * dw

    out = f.values_center * np.exp(1j * wmin * f.times_center)
    if f.order == 0:
        out[0] *= 0.5
    out = fft.ifft(out, n=N, norm="forward")

    if f.order > 0:
        out *= np.exp(1j * np.arange(N) * dw * f.times_center[0])

        for kw, w in enumerate(w_samples):

            alp_values = np.exp(1j * w * f.times_left) * f.values_left
            alp_values += np.exp(1j * w * f.times_right) * f.values_right

            out[kw] += np.sum(f.alpert_weights * alp_values)

    out *= f.delta_t

    return w_samples, out

def inv_ft_to_alpert(wmin, dw, func_vals, M, order):
    N = len(func_vals)
    alpert = AlpertMeshFunction(delta_t=2 * np.pi / (N * dw), M=M, order=order)
    if 2 * alpert.M > N:
        raise ValueError
    a = alpert.a
    M = alpert.M
    dt = alpert.delta_t

    vals_t = fft.fft(func_vals * np.exp(-2j * np.pi * a * np.arange(N) / N), n=N, norm="forward")
    vals_t = vals_t[:M]
    # vals_t = fft.fft(func_vals, n=N, norm="forward")
    # vals_t = vals_t[a:M+a]
    vals_t *= np.exp(-1j * wmin * alpert.times_center) / dt
    alpert.values_center = vals_t

    w_samples = wmin + np.arange(N) * dw
    alpert.values_left = np.empty_like(alpert.times_left, dtype=complex)
    for k, t in enumerate(alpert.times_left):
        alpert.values_left[k] = np.sum(np.exp(-1j * w_samples * t) * func_vals) / (N * dt)

    alpert.values_right = np.empty_like(alpert.times_right, dtype=complex)
    for k, t in enumerate(alpert.times_right):
        alpert.values_right[k] = np.sum(np.exp(-1j * w_samples * t) * func_vals) / (N * dt)

    return alpert
