import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Circle
from scipy.linalg import eig
from PIL import Image
import io

def generate_beachball(strike, dip, rake, size=128):
    """
    生成沙滩球图像
    
    Parameters:
    -----------
    strike : float
        走向角 (degrees)
    dip : float
        倾角 (degrees)
    rake : float
        滑动角 (degrees)
    size : int, optional
        输出图像尺寸，默认为128x128
    
    Returns:
    --------
    numpy.ndarray
        128x128的归一化灰度图像数组
    """
    # 调用bb函数生成沙滩球
    fig, ax = bb([strike, dip, rake], 0, 0, 10, 0, 'red')
    try:
        # Pure in-memory rendering: no Tk window or GUI image resources.
        with io.BytesIO() as buf:
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            buf.seek(0)
            with Image.open(buf) as img:
                resized = img.resize((size, size))
                gray_img = resized.convert('L')
                return np.array(gray_img, dtype=np.float32) / 255.0
    finally:
        fig.clear()

def bb(fm, centerX, centerY, diam, ta, color):
    """
    绘制沙滩球图
    """
    fm = np.array(fm)
    if fm.ndim == 1:
        if len(fm) == 6:
            fm = fm.reshape(1, 6)
        else:
            fm = fm.reshape(1, 3)
    
    ne, n = fm.shape
    if n == 6:
        s1, d1, r1 = [], [], []
        for j in range(ne):
            s, d, r = mij2sdr(fm[j, 0], fm[j, 1], fm[j, 2], fm[j, 3], fm[j, 4], fm[j, 5])
            s1.append(s)
            d1.append(d)
            r1.append(r)
        s1 = np.array(s1)
        d1 = np.array(d1)
        r1 = np.array(r1)
    else:
        s1 = fm[:, 0]
        d1 = fm[:, 1]
        r1 = fm[:, 2]
    
    d2r = np.pi / 180
    
    if isinstance(centerY, (int, float)):
        ampy = np.cos(centerY * d2r)
    else:
        ampy = np.cos(np.mean(centerY) * d2r)
    
    mech = np.zeros(ne)
    j = np.where(r1 > 180)[0]
    r1[j] = r1[j] - 180
    mech[j] = 1
    j = np.where(r1 < 0)[0]
    r1[j] = r1[j] + 180
    mech[j] = 1
    
    s2, d2, r2 = AuxPlane(s1, d1, r1)
    
    for ev in range(ne):
        S1 = s1[ev]
        D1 = d1[ev]
        S2 = s2[ev]
        D2 = d2[ev]
        CX = centerX[ev] if isinstance(centerX, (np.ndarray, list)) else centerX
        CY = centerY[ev] if isinstance(centerY, (np.ndarray, list)) else centerY
        D = diam[ev] if isinstance(diam, (np.ndarray, list)) else diam
        M = mech[ev]
        
        if M > 0:
            P = 2
        else:
            P = 1
        
        if D1 >= 90:
            D1 = 89.9999
        if D2 >= 90:
            D2 = 89.9999
        
        phi = np.arange(0, np.pi + 0.01, 0.01)
        d = 90 - D1
        m = 90
        l1 = np.sqrt(d**2 / (np.sin(phi)**2 + np.cos(phi)**2 * d**2 / m**2))
        
        d = 90 - D2
        m = 90
        l2 = np.sqrt(d**2 / (np.sin(phi)**2 + np.cos(phi)**2 * d**2 / m**2))
        
        inc = 1
        X1, Y1 = pol2cart(phi + S1 * d2r, l1)
        
        if P == 1:
            lo = S1 - 180
            hi = S2
            if lo > hi:
                inc = -inc
            th1 = np.arange(S1 - 180, S2 + inc, inc)
            Xs1, Ys1 = pol2cart(th1 * d2r, 90 * np.ones(len(th1)))
            X2, Y2 = pol2cart(phi + S2 * d2r, l2)
            th2 = np.arange(S2 + 180, S1 - inc, -inc)
        else:
            hi = S1 - 180
            lo = S2 - 180
            if lo > hi:
                inc = -inc
            th1 = np.arange(hi, lo - inc, -inc)
            Xs1, Ys1 = pol2cart(th1 * d2r, 90 * np.ones(len(th1)))
            X2, Y2 = pol2cart(phi + S2 * d2r, l2)
            X2 = np.flip(X2)
            Y2 = np.flip(Y2)
            th2 = np.arange(S2, S1 + inc, inc)
        
        Xs2, Ys2 = pol2cart(th2 * d2r, 90 * np.ones(len(th2)))
        
        X = np.concatenate((X1, Xs1, X2, Xs2))
        Y = np.concatenate((Y1, Ys1, Y2, Ys2))
        
        if D > 0:
            X = ampy * X * D / 90 + CY
            Y = Y * D / 90 + CX
            
            fig = Figure(figsize=(5, 5))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 绘制白色背景圆
            circle = Circle((CX, CY), D, color='white')
            ax.add_patch(circle)
            
            # 绘制沙滩球
            polygon = Polygon(np.column_stack((Y, X)), closed=True, color=color)
            ax.add_patch(polygon)
            
            # 绘制轮廓
            outline = Circle((CX, CY), D, fill=False, color='black', linewidth=0.5)
            ax.add_patch(outline)
            
            ax.set_xlim(CX - D - 5, CX + D + 5)
            ax.set_ylim(CY - D - 5, CY + D + 5)
            
            return fig, ax

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def AuxPlane(s1, d1, r1):
    r2d = 180 / np.pi
    d2r = np.pi / 180
    
    z = (s1 + 90) * d2r
    z2 = d1 * d2r
    z3 = r1 * d2r
    
    sl1 = -np.cos(z3) * np.cos(z) - np.sin(z3) * np.sin(z) * np.cos(z2)
    sl2 = np.cos(z3) * np.sin(z) - np.sin(z3) * np.cos(z) * np.cos(z2)
    sl3 = np.sin(z3) * np.sin(z2)
    
    strike, dip = strikedip(sl2, sl1, sl3)
    
    n1 = np.sin(z) * np.sin(z2)
    n2 = np.cos(z) * np.sin(z2)
    n3 = np.cos(z2)
    h1 = -sl2
    h2 = sl1
    
    z_val = h1 * n1 + h2 * n2
    denominator = np.sqrt(h1**2 + h2**2)
    z_val = np.divide(
        z_val,
        denominator,
        out=np.zeros_like(z_val, dtype=float),
        where=denominator > np.finfo(float).eps,
    )
    # Floating-point roundoff can produce 1+epsilon and make arccos return NaN.
    z_val = np.arccos(np.clip(z_val, -1.0, 1.0))
    
    rake = np.zeros_like(strike)
    j = np.where(sl3 > 0)[0]
    rake[j] = z_val[j] * r2d
    j = np.where(sl3 <= 0)[0]
    rake[j] = -z_val[j] * r2d
    
    return strike, dip, rake

def strikedip(n, e, u):
    r2d = 180 / np.pi
    
    j = np.where(u < 0)[0]
    n[j] = -n[j]
    e[j] = -e[j]
    u[j] = -u[j]
    
    strike = np.arctan2(e, n) * r2d
    strike = strike - 90
    strike = np.mod(strike, 360)
    
    x = np.sqrt(n**2 + e**2)
    dip = np.arctan2(x, u) * r2d
    
    return strike, dip

def mij2sdr(mxx, myy, mzz, mxy, mxz, myz):
    a = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]])
    D, V = eig(a)
    
    idx = D.argsort()
    D = D[idx]
    V = V[:, idx]
    
    V[1:, :] = -V[1:, :]
    
    new_order = [2, 0, 1]
    D = D[new_order]
    V = V[:, new_order]
    V = np.vstack((V[1, :], V[2, :], V[0, :]))
    
    AE = (V[:, 0] + V[:, 2]) / np.sqrt(2.0)
    AN = (V[:, 0] - V[:, 2]) / np.sqrt(2.0)
    AER = np.sqrt(np.sum(AE**2))
    ANR = np.sqrt(np.sum(AN**2))
    AE = AE / AER
    AN = AN / ANR
    
    if AN[2] <= 0:
        AN1 = AN
        AE1 = AE
    else:
        AN1 = -AN
        AE1 = -AE
    
    ft, fd, fl = TDL(AN1, AE1)
    strike = 360 - ft
    dip = fd
    rake = 180 - fl
    
    return strike, dip, rake

def TDL(AN, BN):
    XN, YN, ZN = AN
    XE, YE, ZE = BN
    AAA = 1.0e-06
    CON = 57.2957795
    
    if abs(ZN) < AAA:
        FD = 90.0
        AXN = abs(XN)
        if AXN > 1.0:
            AXN = 1.0
        FT = np.arcsin(AXN) * CON
        ST = -XN
        CT = YN
        
        if ST >= 0 and CT < 0:
            FT = 180 - FT
        if ST < 0 and CT <= 0:
            FT = 180 + FT
        if ST < 0 and CT > 0:
            FT = 360 - FT
        
        FL = np.arcsin(abs(ZE)) * CON
        SL = -ZE
        
        if abs(XN) < AAA:
            CL = XE / YN
        else:
            CL = -YE / XN
        
        if SL >= 0 and CL < 0:
            FL = 180 - FL
        if SL < 0 and CL <= 0:
            FL = FL - 180
        if SL < 0 and CL > 0:
            FL = -FL
    else:
        if -ZN > 1.0:
            ZN = -1.0
        FDH = np.arccos(-ZN)
        FD = FDH * CON
        SD = np.sin(FDH)
        
        if SD == 0:
            return 0, 0, 0
        
        ST = -XN / SD
        CT = YN / SD
        SX = abs(ST)
        if SX > 1.0:
            SX = 1.0
        FT = np.arcsin(SX) * CON
        
        if ST >= 0 and CT < 0:
            FT = 180 - FT
        if ST < 0 and CT <= 0:
            FT = 180 + FT
        if ST < 0 and CT > 0:
            FT = 360 - FT
        
        SL = -ZE / SD
        SX = abs(SL)
        if SX > 1.0:
            SX = 1.0
        FL = np.arcsin(SX) * CON
        
        if ST == 0:
            CL = XE / CT
        else:
            XXX = YN * ZN * ZE / SD**2 + YE
            CL = -SD * XXX / XN
            if CT == 0:
                CL = YE / ST
        
        if SL >= 0 and CL < 0:
            FL = 180 - FL
        if SL < 0 and CL <= 0:
            FL = FL - 180
        if SL < 0 and CL > 0:
            FL = -FL
    
    return FT, FD, FL

# 使用示例
if __name__ == "__main__":
    # 示例：生成一个沙滩球图像
    strike = 345   # 走向角
    dip = 59      # 倾角
    rake = -77     # 滑动角
    
    beachball_img = generate_beachball(strike, dip, rake)
    print(f"生成的图像形状: {beachball_img.shape}")
    print(f"像素值范围: [{beachball_img.min():.3f}, {beachball_img.max():.3f}]")
    
    # 可以显示图像
    plt.imshow(beachball_img, cmap='gray')
    plt.title(f'Strike: {strike}°, Dip: {dip}°, Rake: {rake}°')
    plt.axis('off')
    plt.show()