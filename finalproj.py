from AstroLi import *

def radec2UnitVec(ra, dec):
    # takes obs ra and dec, and returns geocentric unit vectors

    z = 1

    h = z/np.tan(dec.rad)

    x = h/np.sqrt(1 + np.tan(ra.rad)**2) 
    y = np.sqrt(h**2 - x**2)

    r = Vector3D(x, y, z)
    r = r.unitVec()

    return r

def cunninghamFrame(p1, p2, p3):

    xi = p1

    eta = p1.cross(p3.cross(p1))
    eta = eta.unitVec()

    zeta = xi.cross(eta)

    return xi.unitVec(), eta.unitVec(), zeta.unitVec()


def cunninghamObs(p1, p2, p3, xi, eta, zeta):

    rho1 = Vector3D(p1.dot(xi), p1.dot(eta), p1.dot(zeta))
    rho2 = Vector3D(p2.dot(xi), p2.dot(eta), p2.dot(zeta))
    rho3 = Vector3D(p3.dot(xi), p3.dot(eta), p3.dot(zeta))
    
    return rho1.unitVec(), rho2.unitVec(), rho3.unitVec()

def earthSunVect(jd):
    pass

def jd2M(jd, mu, k_orbit):
    
    L = 100.46435*np.pi/180
    jd_2000 = 2451545.0

    n = np.sqrt(mu*k_orbit.a**(-3))

    M_2000 = L - k_orbit.O.rad + k_orbit.w.rad
    M = (M_2000 + n*(jd - jd_2000))%(2*np.pi)

    return M

def M2E(M, k_orbit):

    tol = 1e-4
    buf = 1
    N = 100

    E = np.linspace(0, 2*np.pi, N)
    M_temp = E - k_orbit.e*np.sin(E)

    done = False

    while not done:
        min_indx = np.argmin(np.abs(M_temp - M))

        closest_approach = np.abs((M_temp - M)[min_indx])

        if closest_approach < tol:
            done = True
            E = E[min_indx]
            break


        if min_indx + buf > N:
            E = np.linspace(E[min_indx - buf], E[N - 1], N)
        elif min_indx - buf < 0:
            E = np.linspace(E[min_indx - buf], E[0], N)
        else:
            E = np.linspace(E[min_indx - buf], E[min_indx + buf], N)

        M_temp = E - k_orbit.e*np.sin(E)

    return E

def jd2f(jd, mu, k_orbit):

    M = jd2M(jd, mu, k_orbit)

    E = M2E(M, k_orbit)

    f = k_orbit.eE2f(E)

    return f

if __name__ == "__main__":

    c = Constants()

    ra = Angle(76.7504965149237393, deg=True)
    dec = Angle(51.8102647002936152, deg=True)
    p1 = radec2UnitVec(ra, dec)

    ra = Angle(76.9265709928143906, deg=True)
    dec = Angle(51.8507984607866774, deg=True)
    p2 = radec2UnitVec(ra, dec)

    ra = Angle(77.2495233320480423, deg=True)
    dec = Angle(51.9434147424934523, deg=True)
    p3 = radec2UnitVec(ra, dec)

    xi, eta, zeta = cunninghamFrame(p1, p2, p3)

    rho1, rho2, rho3 = cunninghamObs(p1, p2, p3, xi, eta, zeta)

    print(rho1, rho2, rho3)

    Earth = KeplerOrbit(1.00000011, 0.01671022, 0.00005, O=-11.26064, w_tilde=102.94719)

    mu = c.G*(1 + c.M_earth/c.M_sun)

    f = jd2f(2452480.5000000000000000, mu, Earth)

    r, v = Earth.orbit2HeliocentricState(mu, f.rad)

    print(r, v)