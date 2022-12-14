import fatpack
import numpy as np
import matplotlib.pyplot as plt
print(fatpack.__version__)


def SN_curve_BS_steel(C2_1, m1, N_knee_2, m2, N_knee_3, m3, N_knee_4, m4):
    # SN curves in DNVGL-RP-C203 are Bilinear (2 segments)
    S_knee = 10**(-(np.log10(N_knee_2)-np.log10(C2_1))/m1)
    curve = fatpack.QuadLinearEnduranceCurve(S_knee)
    curve.Sc = S_knee
    curve.m1 = m1
    curve.m2 = m2
    curve.m3 = m3
    curve.m4 = m4
    curve.Nc = N_knee_2
    curve.Nd = N_knee_2
    curve.Nl = N_knee_3
    curve.Nk = N_knee_4
    return curve


def SN_curve_BS_air(C2, m1, N_knee):
    # d=2
    S_knee = 10**(-(np.log10(N_knee)-np.log10(C2))/m1)
    curve = fatpack.BiLinearEnduranceCurve_flat(S_knee)
    curve.Sc = S_knee
    curve.m1 = m1
    curve.Nc = N_knee
    curve.Nd = N_knee
    return curve


# curve = SN_curve_BS_steel(
#     1.72*10**11, 3.0, 1e6, 5.0, 1e7, 3.0, 5*1e7, 5.0)

curve = SN_curve_BS_air(4.31*10**11, 3.0, 1e7)

N = np.logspace(4, 8, 1000)
S = curve.get_stress(N)


plt.figure(dpi=96)
plt.loglog(N, S)
# plt.title("DNVGL RP-C203 SN curve in air, C")
plt.xlabel("Endurance")
plt.ylabel("Stress range (MPa)")
plt.grid(which='both')


fatigue_limit = curve.get_stress(10**7)
print("Fatigue limit:", fatigue_limit)

loga2 = np.log10(curve.get_endurance(1.0))
print("Intercept of second linear curve log a2:", loga2)
