from AstroLi import *

def radec2UnitVec(ra, dec):
    # takes obs ra and dec, and returns geocentric unit vectors

    z = np.sin(dec.rad)
    y = np.cos(dec.rad)*np.sin(ra.rad)
    x = np.cos(dec.rad)*np.cos(ra.rad)

    r = Vector3D(x, y, z)
    r = r.unitVec()

    return r

def cunninghamFrame(p1, p2, p3):

    xi = p1

    eta = p1.cross(p3.cross(p1))
    eta = eta.unitVec()

    zeta = xi.cross(eta)

    return xi.unitVec(), eta.unitVec(), zeta.unitVec()

def cunninghamTransform(p1, p2, p3, xi, eta, zeta, backward=False):

    x = Vector3D(1, 0, 0)
    y = Vector3D(0, 1, 0)
    z = Vector3D(0, 0, 1)

    M = np.array([[x.dot(xi), y.dot(xi), z.dot(xi)], \
                  [x.dot(eta), y.dot(eta), z.dot(eta)], \
                  [x.dot(zeta), y.dot(zeta), z.dot(zeta)]])


    if backward:
        rho1 = Vector3D(*np.inner(np.transpose(M), (p1).xyz))
        rho2 = Vector3D(*np.inner(np.transpose(M), (p2).xyz))
        rho3 = Vector3D(*np.inner(np.transpose(M), (p3).xyz))
    else:
        rho1 = Vector3D(*np.inner(M, (p1).xyz))
        rho2 = Vector3D(*np.inner(M, (p2).xyz))
        rho3 = Vector3D(*np.inner(M, (p3).xyz))

    return rho1, rho2, rho3


def jd2M(jd, mu, k_orbit):
    
    L = 100.46435*np.pi/180
    jd_2000 = 2451545.0/365.25

    n = np.sqrt(mu*k_orbit.a**(-3))

    M_2000 = L - k_orbit.O.rad - k_orbit.w.rad
    M = (M_2000 + n*(jd - jd_2000))%(2*np.pi)

    return M

def M2E(M, k_orbit):

    tol = 1e-10
    buf = 4
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

    return v_1, v_2, v_3

def orbitalElements(mu, r, v):
    """ Transforms a rotating body state vectors into Keplarian Orbital parameters
    All units are given in AU, yrs, Solar Masses, and degrees
    Inputs:
        mu [float] - Standard gravatational parameter in AU^3 yr^-2
        r [Vector3D] - [x, y, z] state position vector of the orbiting body in AU
        v [Vector3D] - [v_x, v_y, v_z] state velocity vector of the orbiting body in AU/yr
    Outputs:
        a [float] - Semimajor Axis of orbiting body [AU]
        e [float] - Eccentricity of orbit
        i [float] - Inclination of orbit [deg]
        o [float] - Longitude of Accending node [deg]
        w [float] - Argument of Periapsis [deg]
        f [float] - True Anomaly [deg]
    """

    # Catch bad inputs:

    # no mass -> no orbiting!
    if mu == 0:
        print("[ERROR] Standard Gravitational Mass cannot be 0!")
        return None, None, None, None, None, None

    if r.mag() == 0:
        print("[ERROR] r can not be a zero vector!")
        return None, None, None, None, None, None
    
    # if 2/r.mag() <= v.mag()**2/mu:
    #     print("[ERROR] Orbit is no longer elliptical")
    #     return None, None, None, None, None, None
        
    # angular momentum per unit mass
    h = r.cross(v)

    # Calculate a, e, i
    #######################
    a = 1/(2/r.mag() - v.mag()**2/mu)
    e = np.sqrt(1 - h.mag()**2/mu/a)
    i = np.arctan2(np.sqrt(h.x**2 + h.y**2), h.z)

    print("[a] = {:.4f} AU".format(a))
    print("[e] = {:.4f}".format(e))
    print("[i] = {:.4f}°".format(np.degrees(i)%360))
    #######################

    # Case 1: Inclination is 0
    if i == 0:
        # o doesn't make any sense, since it doesn't go above the plane
        o = None
        print("[\u03A9] is undefined")
    else:
        o = np.arctan2(h.x, -h.y)
        print("[\u03A9] = {:.2f}°".format(np.degrees(o)%360))

    # Case 2: Eccentricity is 0
    if e == 0:
        # w doesn't make sense if there is no closest point
        print("[WARNING] e is exactly 0!")
        f = None
        w = None
        print("[f] is undefined")
        print("[\u03C9] is undefined")
        
    else:
        f = np.arctan2(r.dot(v)*h.mag(), h.mag()**2 - mu*r.mag())
        print("[f] = {:.4f}°".format(np.degrees(f)%360))

        # Case 3: Inclinaion is 0, but f is calculated first
        if i == 0:
            w = None
            print("[\u03C9] is undefined")
        else:
            w = np.arctan2(r.z*h.mag()/np.sqrt(h.x**2 + h.y**2), r.x*np.cos(o) + r.y*np.sin(o)) - f
            print("[\u03C9] = {:.4f}°".format(np.degrees(w)%360))
        

    # Returns
    return a, e, i, o, f, w

def findOrbit(O1, O2, O3):

    c = Constants()

    p1 = radec2UnitVec(O1.ra, O1.dec)
    p2 = radec2UnitVec(O2.ra, O2.dec)
    p3 = radec2UnitVec(O3.ra, O3.dec)

    xi, eta, zeta = cunninghamFrame(p1, p2, p3)

    rho1, rho2, rho3 = cunninghamTransform(p1, p2, p3, xi, eta, zeta)

    rho1 = rho1.unitVec()
    rho2 = rho2.unitVec()
    rho3 = rho3.unitVec()

    Earth = KeplerOrbit(1.00000011, 0.01671022, 0.00005, O=-11.26064, w_tilde=102.94719)

    mu = c.G*(1 + c.M_earth/c.M_sun)

    OBL = Angle(23.439291111111, deg=True)

    f = jd2f(O1.jd, mu, Earth)
    R1, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R1 = -Vector3D(*R1.xyz)
    R1 = R1.rotate(-OBL, "x")

    f = jd2f(O2.jd, mu, Earth)
    R2, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R2 = -Vector3D(*R2.xyz)
    R2 = R2.rotate(-OBL, "x")

    f = jd2f(O3.jd, mu, Earth)
    R3, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R3 = -Vector3D(*R3.xyz)
    R3 = R3.rotate(-OBL, "x")

    print("Earth-Sun")
    print(R1, R2, R3)

    R1, R2, R3 = cunninghamTransform(R1, R2, R3, xi, eta, zeta)

    print("Cunningham Earth-Sun")
    print(R1, R2, R3)

    a1 = (O3.jd - O2.jd)/(O3.jd - O1.jd)
    a3 = (O2.jd - O1.jd)/(O3.jd - O1.jd)

    pp2 = (-a1*R1.z + R2.z - a3*R3.z)/rho2.z
    pp3 = (pp2*rho2.y + a1*R1.y - R2.y + a3*R3.y)/a3/rho3.y
    pp1 = (pp2*rho2.x - a3*pp3*rho3.x + a1*R1.x - R2.x + a3*R3.x)/a1

    print('distances')
    print(pp1, pp2, pp3)

    # Geocentric Vectors
    p1 = rho1*pp1
    p2 = rho2*pp2
    p3 = rho3*pp3

    print("Geocentric")
    print(p1, p2, p3)

    p1, p2, p3 = cunninghamTransform(p1, p2, p3, xi, eta, zeta, backward=True)
    R1, R2, R3 = cunninghamTransform(R1, R2, R3, xi, eta, zeta, backward=True)

    # Heliocentric Vectors
    S1 = p1 - R1
    S2 = p2 - R2
    S3 = p3 - R3

    print("Heliocentric")
    print(S1, S2, S3)

    S1 = S1.rotate(OBL, "x")
    S2 = S2.rotate(OBL, "x")
    S3 = S3.rotate(OBL, "x")
    # S1, S2, S3 = cunninghamTransform(S1, S2, S3, xi, eta, zeta, backward=True)

    print("Heliocentric Regular Frame")
    print(S1, S2, S3)

    v_1, v_2, v_3 = r2V(O1, O2, O3, S1, S2, S3, mu)

    

    print("Speeds")
    print(v_3, v_2, v_1)

    print("########## Orbit 1")
    orbitalElements(mu, S1, v_1)

    print("########## Orbit 2")
    orbitalElements(mu, S2, v_2)

    print("########## Orbit 3")
    orbitalElements(mu, S3, v_3)



if __name__ == "__main__":

    # RA, Dec, jd, Mag
    O1 = Observation(76.7504965149237393, 51.8102647002936152, 2452465.5000000000000000, 24.658)
    O2 = Observation(76.9265709928143906, 51.8507984607866774, 2452470.5000000000000000, 24.651)
    O3 = Observation(77.2495233320480423, 51.9434147424934523, 2452480.5000000000000000, 24.636)
    O4 = Observation(76.8571921038517445, 51.8341002906526711, 2452468.5000000000000000, 24.654)
    O5 = Observation(77.0607715091441463, 51.8860672841479200, 2452474.5000000000000000, 24.646)

    C1 = Observation(205.2193073537894463, -13.2218403388722248, 2452465.5, 24.632)
    C2 = Observation(205.9849498113317452, -13.6110792357453203, 2452470.5, 24.603)
    C3 = Observation(207.8799269709847692, -14.4906759088960460, 2452480.5, 24.547)

    findOrbit(O1, O2, O3)

 
