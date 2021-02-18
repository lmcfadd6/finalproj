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

def reverseCunningham(p1, p2, p3, xi, eta, zeta):

    M = np.array([[p1.dot(xi), p1.dot(eta), p1.dot(zeta)], \
                  [p2.dot(xi), p2.dot(eta), p2.dot(zeta)], \
                  [p3.dot(xi), p3.dot(eta), p3.dot(zeta)]])

    rho1 = np.inner(M.T, p1.xyz)
    rho2 = np.inner(M.T, p2.xyz)
    rho3 = np.inner(M.T, p3.xyz)

    return rho1, rho2, rho3

def earthSunVect(jd):
    pass

def jd2M(jd, mu, k_orbit):
    
    L = 100.46435*np.pi/180
    jd_2000 = 2451545.0

    n = np.sqrt(mu*k_orbit.a**(-3))

    M_2000 = L - k_orbit.O.rad - k_orbit.w.rad
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

def myfunctionF(mu, r_0, t, t_0):

    return 1 - 0.5*mu/r_0.mag()**3*(t - t_0)**2

def myfunctionG(mu, r_0, t, t_0):

    return (t - t_0) - 1/6*mu/r_0.mag()**3*(t - t_0)**3

def r2V(O1, O2, O3, R1, R2, R3, mu):

    f = myfunctionF(mu, R2, O3.jd, O2.jd)
    g = myfunctionG(mu, R2, O3.jd, O2.jd)
    v_1 = (R3 - R2*f)*(1/g)

    f = myfunctionF(mu, R3, O1.jd, O3.jd)
    g = myfunctionG(mu, R3, O1.jd, O3.jd)
    v_2 = (R1 - R3*f)*(1/g)

    f = myfunctionF(mu, R1, O2.jd, O1.jd)
    g = myfunctionG(mu, R1, O2.jd, O1.jd)
    v_3 = (R2 - R1*f)*(1/g)

    print("V", v_1, v_2, v_3)

    return v_1

if __name__ == "__main__":

    c = Constants()

    # Switch observation 3 and 4 because sneaky observer made 
    # jd of 1 and 3 the same so that a1 and a3 are divided by 0
    O1 = Observation(76.7504965149237393, 51.8102647002936152, 2452480.5000000000000000, 24.658)
    O2 = Observation(76.9265709928143906, 51.8507984607866774, 2452470.5000000000000000, 24.651)
    O4 = Observation(77.2495233320480423, 51.9434147424934523, 2452480.5000000000000000, 24.636)
    O3 = Observation(76.8571921038517445, 51.8341002906526711, 2452468.5000000000000000, 24.654)
    O5 = Observation(77.0607715091441463, 51.8860672841479200, 2452474.5000000000000000, 24.646)

    p1 = radec2UnitVec(O1.ra, O1.dec)

    p2 = radec2UnitVec(O2.ra, O2.dec)

    p3 = radec2UnitVec(O3.ra, O3.dec)

    xi, eta, zeta = cunninghamFrame(p1, p2, p3)

    rho1, rho2, rho3 = cunninghamObs(p1, p2, p3, xi, eta, zeta)

    print(rho1, rho2, rho3)

    Earth = KeplerOrbit(1.00000011, 0.01671022, 0.00005, O=-11.26064, w_tilde=102.94719)

    mu = c.G*(1 + c.M_earth/c.M_sun)

    f = jd2f(O1.jd, mu, Earth)
    R1, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R1 = Vector3D(*R1.xyz)

    f = jd2f(O2.jd, mu, Earth)
    R2, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R2 = Vector3D(*R2.xyz)

    f = jd2f(O3.jd, mu, Earth)
    R3, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R3 = Vector3D(*R3.xyz)

    R1, R2, R3 = cunninghamObs(R1, R2, R3, xi, eta, zeta)

    a1 = (O3.jd - O2.jd)/(O3.jd - O1.jd)
    a3 = (O2.jd - O1.jd)/(O3.jd - O1.jd)

    pp2 = (-a1*R1.z + R2.z - a3*R3.z)/rho2.z
    pp3 = (pp2*rho2.y + a1*R1.y - R2.y + a3*R3.y)/a3/rho3.y
    pp1 = (pp2*rho2.x - a3*pp3*rho3.x + a1*R1.x - R2.x +a3*R3.x)/a3

    # Geocentric Vectors
    p1 = rho1*pp1
    p2 = rho2*pp2
    p3 = rho3*pp3

    print(p1, p2, p3)

    # Heliocentric Vectors
    S1 = p1 + R1
    S2 = p2 + R2
    S3 = p3 + R3

    print(S1, S2, S3)

    v = r2V(O1, O2, O3, S1, S2, S3, mu)

