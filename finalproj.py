##### Imports

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from AstroLi import *

##### Constants
c = Constants()
Earth = KeplerOrbit(1.00000011, 0.01671022, 0.00005, O=-11.26064, w_tilde=102.94719)

def radec2UnitVec(ra, dec):

    ''' takes observevd ra and dec, and returns geocentric unit vectors
    Inputs:
    ra [Angle Obj] - Right Ascension of object
    dec [Angle Obj] - Declination of object

    Retruns:
    r [Vector3D] - Unit vector of object direction
    '''

    z = np.sin(dec.rad)
    y = np.cos(dec.rad)*np.sin(ra.rad)
    x = np.cos(dec.rad)*np.cos(ra.rad)

    r = Vector3D(x, y, z)
    # r = r.unitVec()

    return r

def cunninghamFrame(p1, p2, p3):

    xi = p1

    eta = p1.cross(p3.cross(p1))
    eta = eta.unitVec()

    zeta = xi.cross(eta)

    return xi.unitVec(), eta.unitVec(), zeta.unitVec()

def cunninghamTransform(p1, p2, p3, xi, eta, zeta, backward=False):
    ''' Transforms a set of vectors into the cunningham frame

    Inputs:
    p1, p2, p3 [Vector Objs] - Three vectors to transform
    xi, eta, zeta [Vector Objs] - Three unit vectors defining the cunningham frame

    Outputs:
    rho1, rho2, rho3 [Vector Objs] - p1, p2, p3 rotated into the cunningham frame
    '''

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
    ''' Converts a julian day into mean anomaly

    Inputs:
    jd [float] - julian day
    mu [float] - standard gravitational parameter
    k_orbit [KeplerOrbit Obj] - Orbit of object to convert

    Outputs:
    M [float] - Mean anomaly at given jd in radians
    '''
    
    L = np.radians(100.46435)
    jd_2000 = 2451545.0

    n = np.sqrt(mu/k_orbit.a**3)

    M_2000 = L - k_orbit.O.rad - k_orbit.w.rad
    M = (M_2000 + n*(jd - jd_2000)/c.days_per_year)%(2*np.pi)

    return M

def M2E(M, k_orbit):
    ''' Converts a mean anomaly into eccentric anomaly

    Inputs:
    M [float] - Mean anomaly in radians
    k_orbit [KeplerOrbit Obj] - Orbit of object to convert

    Outputs:
    E [float] - Eccentric anomaly in radians
    '''

    tol = 1e-10
    buf = 4
    N = 1000

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


        if min_indx + buf >= N:
            E = np.linspace(E[min_indx - buf], E[N - 1], N)
        elif min_indx - buf < 0:
            E = np.linspace(E[min_indx - buf], E[0], N)
        else:
            E = np.linspace(E[min_indx - buf], E[min_indx + buf], N)

        M_temp = E - k_orbit.e*np.sin(E)

    return E

def jd2f(jd, mu, k_orbit, verbose=False):
    ''' Converts a julian date to true anomaly in 3 steps
    see helper functions for more detail

    Inputs: 
    jd [float] - julian day
    mu [float] - standard parameter
    k_orbit [KeplerOrbit Obj] - Orbit of object to convert

    Outputs:
    f [Angle Obj] - true anomaly
    '''

    M = jd2M(jd, mu, k_orbit)

    if verbose:
        print("M", np.degrees(M))

    E = M2E(M, k_orbit)

    f = k_orbit.eE2f(E)

    return f

def myfunctionF(mu, r_0, t, t_0):
    ''' f in equations to estimate velocity from position vectors

    Inputs:
    mu [float] - standard gravitational parameter
    r_0 [Vector3D Obj] - initial position vector of object
    t [float] - final julian day of object
    t_0 [float] - initial julian day of object

    Outputs:
    f(t, t_0) [float] 
    '''

    sigma = mu/(r_0.mag())**3
    
    return 1 - 0.5*sigma*((t - t_0)/c.days_per_year)**2

def myfunctionG(mu, r_0, t, t_0):
    ''' g in equations to estimate velocity from position vectors

    Inputs:
    mu [float] - standard gravitational parameter
    r_0 [Vector3D Obj] - initial position vector of object
    t [float] - final julian day of object
    t_0 [float] - initial julian day of object

    Outputs:
    g(t, t_0) [float] 
    '''    

    sigma = mu/(r_0.mag())**3

    return ((t - t_0)/c.days_per_year) - 1/6*sigma*((t - t_0)/c.days_per_year)**3

def r2V(O1, O2, O3, R1, R2, R3, mu):
    ''' Estimates velocity vectors from position vectors for three observations

    Inputs:
    O1, O2, O3 [Observation Objs] - Observations to predict velocities of
    R1, R2, R3 [Vector3D Objs] - Position vectors of the three observations

    Outputs:
    v_1, v_2, v_3 [Vector3D Obj] - Three velocity vector estimates of object
    '''

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

def findOrbit(O1, O2, O3, factor=1):
    ''' Predict orbit of an object given 3 observation objects

    Inputs:
    O1, O2, O3 [Observation Objs] - Observations to predict from
    factor [float] - Optional fudge factor, multiplies sector ratios by this number, useful 
    for seeing how variable solutions are

    Returns:
    O1, O2, O3 [Observation Obj] - Observations with orbits included (call O1.orbit for orbit)
    O1.orbit.e, O2.orbit.e, O3.orbit.e [float] - Eccentricities of orbit (useful for bugfixing unbound orbits)
    '''

    # Unit vectors of observatins
    p1 = radec2UnitVec(O1.ra, O1.dec)
    p2 = radec2UnitVec(O2.ra, O2.dec)
    p3 = radec2UnitVec(O3.ra, O3.dec)

    print("Obsevation Unit Vectors:")
    print("####################")
    print(p1)
    print(p2)
    print(p3)

    # Cunningham frame
    xi, eta, zeta = cunninghamFrame(p1, p2, p3)

    print("")
    print("Cunningham Frame Vectors")
    print("####################")
    print(xi)
    print(eta)
    print(zeta)

    # Rotate to cunningham frame
    rho1, rho2, rho3 = cunninghamTransform(p1, p2, p3, xi, eta, zeta)

    # Make sure these are unit vectors
    rho1 = rho1.unitVec()
    rho2 = rho2.unitVec()
    rho3 = rho3.unitVec()

    print("")
    print("Observation Unit Vectors (Cunningham Frame)")
    print("####################")
    print(rho1)
    print(rho2)
    print(rho3)

    mu = c.G*(1 + c.M_earth/c.M_sun)

    OBL = Angle(23.439291111111, deg=True)

    # Find Earth-Sun vector at given observation time
    f = jd2f(O1.jd, mu, Earth)
    R1, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R1 = Vector3D(*R1.xyz)
    R1 = -R1.rotate(-OBL, "x")

    f = jd2f(O2.jd, mu, Earth)
    R2, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R2 = Vector3D(*R2.xyz)
    R2 = -R2.rotate(-OBL, "x")

    f = jd2f(O3.jd, mu, Earth)
    R3, _ = Earth.orbit2HeliocentricState(mu, f.rad)
    R3 = Vector3D(*R3.xyz)
    R3 = -R3.rotate(-OBL, "x")

    print("")
    print("Earth-Sun Vectors")
    print("####################")
    print(R1)
    print(R2)
    print(R3)

    # Transform to cunningham
    R1, R2, R3 = cunninghamTransform(R1, R2, R3, xi, eta, zeta)

    print("")
    print("Earth-Sun Vectors (Cunningham Frame)")
    print("####################")
    print(R1)
    print(R2)
    print(R3)

    # Sector ratios
    a1 = (O3.jd - O2.jd)/(O3.jd - O1.jd)*factor
    a3 = (O2.jd - O1.jd)/(O3.jd - O1.jd)*factor

    print("")
    print("Sector Ratios")
    print("####################")
    print("a1 = {:.2f}".format(a1))
    print("a3 = {:.2f}".format(a3))

    if rho2.z <= 1e-3:
        message = "(close to zero)"
    print("\u03BD_2 = {:.6E} {:}".format(rho2.z, message))

    # Magnitudes of observation vectors
    pp2 = (-a1*R1.z + R2.z - a3*R3.z)/rho2.z
    pp3 = (pp2*rho2.y + a1*R1.y - R2.y + a3*R3.y)/a3/rho3.y
    pp1 = (pp2*rho2.x - a3*pp3*rho3.x + a1*R1.x - R2.x + a3*R3.x)/a1

    print("")
    print('Distances')
    print("####################")
    print("{:.4f} AU".format(pp1))
    print("{:.4f} AU".format(pp2))
    print("{:.4f} AU".format(pp3))

    # Geocentric Vectors
    p1 = rho1*pp1
    p2 = rho2*pp2
    p3 = rho3*pp3

    p1, p2, p3 = cunninghamTransform(p1, p2, p3, xi, eta, zeta, backward=True)
    R1, R2, R3 = cunninghamTransform(R1, R2, R3, xi, eta, zeta, backward=True)

    # Heliocentric Vectors
    S1 = p1 - R1
    S2 = p2 - R2
    S3 = p3 - R3

    S1 = S1.rotate(OBL, "x")
    S2 = S2.rotate(OBL, "x")
    S3 = S3.rotate(OBL, "x")

    # Store these vectors in observation objects
    O1.geoVect = p1
    O2.geoVect = p2
    O3.geoVect = p3

    O1.sunVect = S1
    O2.sunVect = S2
    O3.sunVect = S3

    print("")
    print('Geocentric Vectors')
    print("####################")
    print(p1)
    print(p2)
    print(p3)

    print("")
    print('Heliocentric Vectors')
    print("####################")
    print(S1)
    print(S2)
    print(S3)


    mu = c.G

    # approximate velocity
    v_1, v_2, v_3 = r2V(O1, O2, O3, S1, S2, S3, mu)

    print("")
    print('Velocity Vectors')
    print("####################")
    print(v_1)
    print(v_2)
    print(v_3)

    print("")
    print("########## Orbit 1")
    a, e, i, o, f, w = orbitalElements(mu, S1, v_1)
    Orbit1 = KeplerOrbit(a, e, np.degrees(i), O=np.degrees(o), w=np.degrees(w))

    print("")
    print("########## Orbit 2")
    a, e, i, o, f, w = orbitalElements(mu, S2, v_2)
    Orbit2 = KeplerOrbit(a, e, np.degrees(i), O=np.degrees(o), w=np.degrees(w))

    print("")
    print("########## Orbit 3")
    a, e, i, o, f, w = orbitalElements(mu, S3, v_3)
    Orbit3 = KeplerOrbit(a, e, np.degrees(i), O=np.degrees(o), w=np.degrees(w))

    # Store orbit into observation objects
    O1.orbit = Orbit1
    O2.orbit = Orbit2
    O3.orbit = Orbit3

    return O1, O2, O3, O1.orbit.e, O2.orbit.e, O3.orbit.e

def plotOrbits(O1, O2, O3):
    """ Plots orbits from observations along with mean orbit and other planets

    Inputs:
    O1, O2, O3 [Observation Objs] - Observation objects with orbits included (run findOrbit() first!)

    """

    # If mass of the planets is 0:
    c = Constants()
    mu_sun = c.G

    # Define Keplar Orbit Objects from table
    #(a, e, i, O=None, w=None, f=None, w_tilde=None)
    Mercury = KeplerOrbit(0.38709893, 0.20563069, 7.00487, O=48.33167, w_tilde=77.45645)
    Venus = KeplerOrbit(0.72333199, 0.00677323, 3.39471, O=76.68069, w_tilde=131.53298)
    Mars = KeplerOrbit(1.52366231, 0.09341233, 1.85061, O=49.57854, w_tilde=336.04084)
    Jupiter = KeplerOrbit(5.20336301, 0.04839266, 1.30530, O=100.55615, w_tilde=14.75385)
    Saturn = KeplerOrbit(9.53707032, 0.05415060, 2.48446, O=113.71504, w_tilde=92.43194)
    Uranus = KeplerOrbit(19.19126393, 0.04716771, 0.76986, O=74.22988, w_tilde=170.96424)
    Neptune = KeplerOrbit(30.06896348, 0.00858587, 1.76917, O=131.72169, w_tilde=44.97135)
    Pluto = KeplerOrbit(39.48168677, 0.24880766, 17.14175, O=110.30347, w_tilde=224.06676)

    # https://ssd.jpl.nasa.gov/?sb_elem
    # Epoch J2000
    # Halley gives q instead of a
    # Ceres = KeplerOrbit(2.7653485, 0.07913825,  10.58682,  w=72.58981,  O=80.39320)
    # Halley = KeplerOrbit(0.58597811/(1-0.96714291), 0.96714291, 162.26269, w=111.33249, O=58.42008)


    mean = KeplerOrbit(np.mean([O1.a, O2.a, O3.a]), \
                        np.mean([O1.e, O2.e, O3.e]), \
                        np.mean([O1.i.deg, O2.i.deg, O3.i.deg]), \
                        O=np.mean([O1.O.deg, O2.O.deg, O3.O.deg]), \
                        w=np.mean([O1.w.deg, O2.w.deg, O3.w.deg]))

    # Extra information for plotting
    planets = [Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, O1, O2, O3, mean]
    colours = ["#947876", "#bf7d26", "#479ef5", "#fa0707", "#c79e0a", "#bdba04", "#02edd6", "#2200ff", "#a3986c", "#000000", "#000000", "#000000", "#fa0707"]
    names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",  "Orbit1", "Orbit2", "Orbit3", "Mean Orbit"]
    NO_OF_PLANETS = len(planets)

    # The following animation code was taken from various stack overflow forums and from the 
    # example given on the matplotlib page
    # https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations

    # Helper function - Organizes all data of an orbit
    def make_planet(n, planet, less=False):
        data_x = []
        data_y = []
        data_z = []


        for f in np.linspace(0, 360, 1800):
            r, v = planet.orbit2HeliocentricState(mu_sun, f)
            data_x.append(r.x)
            data_y.append(r.y)
            data_z.append(r.z)

        data = np.array([data_x, data_y, data_z])
        return data

    # Updates a single planet
    def update(num, data, lines) :

        lines.set_data(data[0:2, num-1:num])
        lines.set_3d_properties(data[2,num-1:num])
        return lines

    # Updates all planets
    def update_all(num, data, lines):

        l = [None]*NO_OF_PLANETS

        for i in range(NO_OF_PLANETS):
            l[i] = update(num, data[i][0], lines[i][0])

        return l

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    n = 100

    data = [None]*NO_OF_PLANETS
    lines = [None]*NO_OF_PLANETS

    # Generate planet lines
    for pp, p in enumerate(planets):
        data[pp] = [make_planet(n, p, less=True)]
        lines[pp] = [ax.plot(data[pp][0][0,0:1], data[pp][0][1,0:1], data[pp][0][2,0:1], \
                c=colours[pp], marker='o', label=names[pp])[0]]


    # Setthe axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X [AU]')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y [AU]')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z [AU]')

    ax.set_title('Kepler Orbits')

    ax.scatter([0], [0], [0], c="y", marker='o', label="Sun")

    # Make cts line orbits
    for pp, planet in enumerate(planets):
        data_x = []
        data_y = []
        data_z = []
        for f in np.linspace(0, 2*np.pi, 1800):
            r, v = planet.orbit2HeliocentricState(mu_sun, f)
            data_x.append(r.x)
            data_y.append(r.y)
            data_z.append(r.z)

        ax.plot(data_x, data_y, data_z, c=colours[pp])

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_all, n, fargs=(data, lines),
                                  interval=50, blit=False)
    plt.legend()
    plt.show()

def asteroidphasefunction(geo_vect, sun_vect):
    """ Calculates the asteroid phase function

    Inputs:
    geo_vect [Vector3D] - Vector from earth to object
    sun_vect [Vector3D] - Vector from sun to object

    Outputs:
    asteroid phase function [float]
    """

    G = 0.15
    A1 = 3.33
    A2 = 1.87
    B1 = 0.63
    B2 = 1.22

    alpha = Angle(np.arccos(geo_vect.dot(sun_vect)/geo_vect.mag()/sun_vect.mag()))

    phi_1 = np.exp(-A1*np.tan(alpha.rad/2)**B1)
    phi_2 = np.exp(-A2*np.tan(alpha.rad/2)**B2)
    return 2.5*np.log10((1 - G)*phi_1 + G*phi_2)

def calcMags(O1, O2, O3):

    """ Calculates absolute magnitude of object from observations (run findOrbits() first!)

    Inputs:
    O1, O2, O3 [Observation Objs] - Observation objects with magnitudes and geo/sun vectors

    Outputs:
    mean_mag [float] - mean absolute magnitude of the three guesses
    O1, O2, O3 [Observation Objs] - Observation Objects with their absolute magnitude included (call O.H)
    """

    print("")
    print("Absolute Magnitudes")
    print("####################")    

    for oo, O in enumerate([O1, O2, O3]):

        M = 5*np.log10(O.geoVect.mag()*O.sunVect.mag())

        O.H = O.mag - M + asteroidphasefunction(O.geoVect, O.sunVect)

        print("Magnitude {:} = {:.4f}".format(oo + 1, O.H))

    mean_mag = np.mean([O1.H, O2.H, O3.H])

    return mean_mag, O1, O2, O3

def diameterCalc(mag, albedo):
    """ Calculates diameter from formula given in notes

    Inputs:
    mag [float] - Absolute magnitude of object
    albedo [float] - albedo of object

    Outputs:
    diameter [float] - Diameter in kilometers
    """

    return 1329*10**(-mag/5)*albedo**(-0.5)

def genRaDecPlot(O1, O2, O3, mag):
    """ Generates RA and Dec plots FOR THIS PROJECT ONLY (change hardcoded points below if needed)

    Inputs:
    O1, O2, O3 [Observation Objs] - Observation Objs (run findOrbit() first!)
    mag [float] - absolute magnitude of object
    """


    mu = c.G
    jd = 2452465.5
    OBL = Angle(23.439291111111, deg=True)

    ra_list = []
    dec_list = []
    jd_list = []
    mag_list = []
    sun_dist_list = []
    earth_dist_list = []
    sun_earth_list = []

    rx = []
    ry = []
    sx = []
    sy = []

    H = mag

    O = O3

    for jj in np.linspace(0, 120, 121):
        
        jy = (jd + jj)

        # get sun-earth vector at a given jd
        f = jd2f(jy, mu, Earth)
        R, _ = Earth.orbit2HeliocentricState(mu, f.rad)
        R = Vector3D(*R.xyz)

        # # get sun-obj vector at a given jd 
        f = jd2f(jy, mu, O.orbit)
        S, _ = O.orbit.orbit2HeliocentricState(mu, f.rad)
        S = Vector3D(*S.xyz)

        # earth-obj vector
        V = S - R
        V = V.rotate(-OBL, "x")


        ra, dec = cart2Radec(V)

        ra_sun, dec_sun = cart2Radec(S)

        m = H + 5*np.log10(S.mag()*V.mag()) - asteroidphasefunction(V, S)

        ra_list.append(ra)
        dec_list.append(dec)
        jd_list.append(jy)
        mag_list.append(m)

        rx.append(R.x)
        ry.append(R.y)
        sx.append(S.x)
        sy.append(S.y)
        sun_dist_list.append(S.mag())
        earth_dist_list.append(V.mag())
        sun_earth_list.append(R.mag())


    plt.subplot(2, 1, 1)
    plt.plot(jd_list, mag_list)
    plt.xlabel("Julian Day [days]")
    plt.ylabel("Apparent Magnitude")
    plt.scatter(np.array([2452465.5, 2452470.5, 2452480.5, 2452468.5, 2452474.5]), \
                [24.658, 24.651, 24.636, 24.654, 24.646])
    plt.subplot(2, 1, 2)
    plt.plot(ra_list, dec_list)
    plt.xlabel("Right Ascension [deg]")
    plt.ylabel("Declination [deg]")
    plt.scatter([76.7504965149237393, 76.9265709928143906, 77.2495233320480423, 76.8571921038517445, 77.0607715091441463], \
                [51.8102647002936152, 51.8507984607866774, 51.9434147424934523, 51.8341002906526711, 51.8860672841479200])

    plt.show()


if __name__ == "__main__":

    # RA, Dec, jd, Mag
    O1 = Observation(76.7504965149237393, 51.8102647002936152, 2452465.5,  24.658)
    O2 = Observation(76.9265709928143906, 51.8507984607866774, 2452470.5,  24.651)
    O3 = Observation(77.2495233320480423, 51.9434147424934523, 2452480.5141193,  24.636)
    O4 = Observation(76.8571921038517445, 51.8341002906526711, 2452468.5,  24.654)
    O5 = Observation(77.0607715091441463, 51.8860672841479200, 2452474.5,  24.646)

    # Cole's Observations
    C1 = Observation(205.2193073537894463, -13.2218403388722248, 2452465.5, 24.632)
    C2 = Observation(205.9849498113317452, -13.6110792357453203, 2452470.5, 24.603)
    C3 = Observation(207.8799269709847692, -14.4906759088960460, 2452480.5, 24.547)

    print("Observations")
    print("####################")
    print(O1)
    print(O2)
    print(O3)    

    O1, O2, O3, e1, e2, e3 = findOrbit(O1, O2, O3, factor=1)

    ### Generate "fudge" factor plot
    # f_list = []
    # elist1 = []
    # elist2 = []
    # elist3 = []

    # for factor in np.linspace(0.99, 1.01, 1000):
    #     O1, O2, O3, e1, e2, e3 = findOrbit(O1, O2, O3, factor)
    #     f_list.append(factor)
    #     elist1.append(e1)
    #     elist2.append(e2)
    #     elist3.append(e3)

    # plt.semilogy(f_list, elist1, label='Orbit 1')
    # plt.semilogy(f_list, elist2, label='Orbit 2')
    # plt.semilogy(f_list, elist3, label='Orbit 3')
    # plt.hlines(1, 0.99, 1.01)
    # plt.axis((0.99, 1.01, 0.1, 10))
    # plt.xlabel('"Fudge" Factor')
    # plt.ylabel("Eccentricity, e")
    # plt.legend()
    # plt.show()
    # exit()
    # plotOrbits(O1.orbit, O2.orbit, O3.orbit)
    
    mag, _, _, _ = calcMags(O1, O2, O3)

    ALBEDO_COMET = 0.04
    ALBEDO_ASTEROID = 0.10

    d = diameterCalc(mag, ALBEDO_COMET)

    print("")
    print("Diameter of object")
    print("####################")
    print("{:.2f} km".format(d))

    genRaDecPlot(O1, O2, O3, mag)