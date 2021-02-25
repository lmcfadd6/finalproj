
import numpy as np

class Constants:
    """ An object defining constants
    """ 

    def __init__(self):

        # Gravitational Constant
        self.G = 4*np.pi**2 #AU^3 yr^-2 M_sun^-1

        # Astronomical Unit in meters
        self.AU = 1.496e+11 # m

        # Seconds in a year
        self.yr = 31557600 #s

        # Solar mass in kg
        self.M_sun = 1.989e30 #kg

        # Earth mass in kg
        self.M_earth = 3.0404327497692654e-06*self.M_sun #kg

        # Days per year
        self.days_per_year = 365.2568983263281 #days



class Vector3D:
    """ Basic function defining 3D cartesian vectors in [x, y, z] form

        Example:
            u = Vector3D(0, 1, 2)
            v = Vector3D(1, 1, 1)
            u is a vector [0, 1, 2]
            v is a vector [1, 1, 1]

    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]
        self.h = (x*x + y*y)**0.5
        self.r = (x*x + y*y + z*z)**0.5

        self.phi =   Angle(np.arctan2(self.y, self.x))
        self.theta = Angle(np.arctan2(self.h, self.z))

    def __add__(self, other):
        """ Adds two vectors as expected
        Example:
            w = u + v
            w is a vector [1, 2, 2]
        
        """
        result = Vector3D(self.x + other.x, \
                          self.y + other.y, \
                          self.z + other.z)
        return result

    def __sub__(self, other):
        """ Subtracts two vectors as expected
        Example:
            w = u - v
            w is a vector [-1, 0, 1]
        """
        result = Vector3D(self.x - other.x, \
                          self.y - other.y, \
                          self.z - other.z)
        return result


    def __mul__(self, other):
        """ Multipies a vector by a constant
        Example:
            w = u*3
            w is a vector [0, 3, 6]
        """
        result = Vector3D(self.x * other, \
                          self.y * other, \
                          self.z * other)
        return result

    def __neg__(self):

        return Vector3D(-self.x, -self.y, -self.z)

    def __str__(self):
        mag = self.mag()
        if mag == 1.0:
            return 'Unit Vector: [ {:.4f}, {:.4f}, {:.4f}]'.format(self.x, self.y, self.z)
        else:
            return 'Vector: [ {:.4f}, {:.4f}, {:.4f}]'.format(self.x, self.y, self.z)

    def mag(self):
        """ Returns the geometric magnitude of the vector
        """
        result = (self.x**2 + self.y**2 + self.z**2)**0.5
        return result

    def dot(self, other):
        """ Returns the dot product of the vectors
        Example:
            w = u.dot(v)
            w = 3
        """ 
        result = self.x*other.x + self.y*other.y + self.z*other.z

        return result

    def cross(self, other):
        """ Returns the cross product of the vectors
        Example:
            w = u.cross(v)
            w is a vector [-1, 2, -1]
        """

        x = self.y*other.z - self.z*other.y
        y = self.z*other.x - self.x*other.z
        z = self.x*other.y - self.y*other.x

        result = Vector3D(x, y, z)

        return result

    def unitVec(self):

        mag = self.mag()

        x = self.x*(1/mag)
        y = self.y*(1/mag)
        z = self.z*(1/mag)

        result = Vector3D(x, y, z)
        return result

    def rotate(self, ang, axis):
        """ Rotates vector <ang> degrees around an axis
            inputs:
            ang [Angle Obj] - angle to rotate coordinate system by
            axis ["x", "y", or "z"] - axis to rotate vector around 
        """
        
        if axis == "x":
            M = np.array([[1,          0,               0     ], \
                          [0, np.cos(ang.rad), np.sin(ang.rad)], \
                          [0, -np.sin(ang.rad), np.cos(ang.rad)]])
        elif axis == "y":
            M = np.array([[np.cos(ang.rad), 0, -np.sin(ang.rad)], \
                          [0, 1, 0], \
                          [np.sin(ang.rad), 0, np.cos(ang.rad)]])
        elif axis == "z":
            M = np.array([[np.cos(ang.rad), np.sin(ang.rad), 0], \
                          [-np.sin(ang.rad), np.cos(ang.rad), 0], \
                          [0, 0, 1]])
        else:
            print("Unrecognized Axis")
            return None

        vect = np.inner(M, self.xyz)

        return Vector3D(vect[0], vect[1], vect[2])

class Angle:
    """
    Angle object to easilt convert between radians and degrees

    Input:
    ang [float] - angle in radians
    deg [boolean] - if True, angle is given in degrees. If False (default), angle is given in radians
    Example:

        a = Angle(np.pi/2)
        or
        a = Angle(90, deg=True)
    """

    def __init__(self, ang, deg=False):

        if ang is not None:

            if deg:
                self.deg = ang%360
                self.rad = self.deg/180*np.pi
            else:
                self.rad = ang%(2*np.pi)
                self.deg = self.rad*180/np.pi

        else:

            self.deg = None
            self.rad = None

    def __str__(self):

        return "Angle: {:.2f}".format(self.deg)

    def __add__(self, other):

        return Angle(self.rad + other.rad)

    def __sub__(self, other):

        return Angle(self.rad - other.rad)

    def __neg__(self):

        return Angle(-self.rad)

    def unmod(self, deg=False):
        """ Retruns angle in range -180 < x < 180 instead of 0 < x < 360
        """

        if deg:
            if self.deg < 180:
                return self.deg
            return self.deg - 360
        else:
            if self.rad < np.pi:
                return self.rad
            return self.rad - 2*np.pi


    def isNone(self):

        if self.deg is None or self.rad is None:
            return True
        return False

class RightAsc:

    """ Quick object to convert an angle in degrees to a right ascension

    input: 
    angle [float] - angle in degrees
    Example:

        a = RightAsc(90)
        print(a)
        >> Right Ascension: 6.0h 0.0m 0.00s

    """
    def __init__(self, angle):

        self.angle = angle

        total_hours = angle/DEG_PER_HOUR

        self.hour, r = divmod(total_hours, 1)
        self.min, r = divmod(r*60, 1)
        self.sec = r*60


    def __str__(self):

        return "Right Ascension: {:}h {:}m {:.2f}s".format(self.hour, self.min, self.sec)

    def asFloat(self):
        # Returns the angle [deg] instead of right ascention
        
        return self.angle

class Cart:
    """
    A Cart(esian) object takes a geocentric vector defined as [x, y, z] [in meters], 
    and calculates various parameters from it
    """

    def __init__(self, x, y, z):

        try:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        except ValueError:
            print("[WARNING] Cartesian values must be a float")
            return None

        self.xyz = [self.x, self.y, self.z]
        self.h = (x*x + y*y)**0.5
        self.r = (x*x + y*y + z*z)**0.5

        self.phi =   Angle(np.arctan2(self.y, self.x))
        self.theta = Angle(np.arctan2(self.h, self.z))


    def __str__(self):

        return "Cart Obj: x={:.2f} y={:.2f} z={:.2f}".format(self.x, self.y, self.z)

    def __neg__(self):

        return Cart(-self.x, -self.y, -self.z)

    def __sub__(self, other):

        return Cart(self.x - other.x, self.y - other.y, self.z - other.z)

    def rotate(self, ang, axis):
        """ Rotates vector <ang> degrees around an axis
            inputs:
            ang [Angle Obj] - angle to rotate coordinate system by
            axis ["x", "y", or "z"] - axis to rotate vector around 
        """
        
        if axis == "x":
            M = np.array([[1,          0,               0     ], \
                          [0, np.cos(ang.rad), np.sin(ang.rad)], \
                          [0, -np.sin(ang.rad), np.cos(ang.rad)]])
        elif axis == "y":
            M = np.array([[np.cos(ang.rad), 0, -np.sin(ang.rad)], \
                          [0, 1, 0], \
                          [np.sin(ang.rad), 0, np.cos(ang.rad)]])
        elif axis == "z":
            M = np.array([[np.cos(ang.rad), np.sin(ang.rad), 0], \
                          [-np.sin(ang.rad), np.cos(ang.rad), 0], \
                          [0, 0, 1]])
        else:
            print("Unrecognized Axis")
            return None

        vect = np.inner(M, self.xyz)

        return Cart(vect[0], vect[1], vect[2])

##########################
# New content begins here
##########################


class KeplerOrbit:
    """ Define a Keplar orbit given 5 orbital parameters

    Arguments:
    a [Float] - Semimajor axis of the orbit [AU]
    e [Float] - Eccentricity of the orbit
    i [Float] - Inclination of the orbit [Degrees]
    O [Float] - Longitude of the ascending node [Degrees]
    w [Float] - Argument of perihelion [Degrees]
    w_tilde [Float] - Longitude of perihelion [Degrees]

    If w is not given, w may be calculated from O and w_tilde by:
        w_tilde = O + w
    """


    def __init__(self, a, e, i, O=None, w=None, w_tilde=None):

        self.a = a
        self.e = e
        self.i = Angle(i, deg=True)
        self.O = Angle(O, deg=True)
        self.w = Angle(w, deg=True)
        self.w_tilde = Angle(w_tilde, deg=True)

        # If w is not given, calculate through w_tilde = O + w
        if self.w.isNone() and not(self.O.isNone() or self.w_tilde.isNone()):
            self.w = self.w_tilde - self.O


    def __str__(self):

        A = "ORBIT OBJECT: \n"
        A += "[a] = {:.4f} AU \n".format(self.a)
        A += "[e] = {:.4f} \n".format(self.e)
        A += "[i] = {:.4f}° \n".format((self.i.deg)%360)
        A += "[\u03A9] = {:.2f}° \n".format((self.O.deg)%360)
        A += "[\u03C9] = {:.4f}°".format((self.w.deg)%360)

        return A


    def ef2E(self, f, debug=False):
        """ Converts eccentricity, e, and the true anomaly, f, to eccentric
        anomaly E

        Inputs:
        e [Float] - Eccentricity
        f [Float] - True Anomaly [radians]
        debug [Boolean] - If true, prints out return in terminal

        Outputs:
        E [Float] - Eccentric Anomaly [radians]
        """

        f = Angle(f)

        E = np.arctan2(np.tan(f.rad/2),np.sqrt((1 + self.e)/(1 - self.e)))*2

        E = Angle(E)

        if debug:
            print("Eccentric Anomaly = {:.2f} rad".format(E.rad))

        return E

    def eE2f(self, E, debug=False):
        """ Converts eccentricity, e, and eccentric anomaly, E, to the true anomaly, f

        Inputs:
        e [Float] - Eccentricity
        E [Float] - Eccentric Anomaly [radians]
        debug [Boolean] - If true, prints out return in terminal

        Outputs:
        f [Float] - True Anomaly [radians]
        """

        E = Angle(E)

        f = np.arctan2(np.tan(E.rad/2), np.sqrt((1 - self.e)/(1 + self.e)))*2

        f = Angle(f)

        if debug:
            print("True Anomaly = {:.2f} rad".format(f.rad))

        return f

    def orbit2State(self, f, mu):
        """ Takes orbital parameters a, e, and f and the standard gravitational parameter

        Input:
        a [Float] - Semimajor axis of the orbit [AU]
        e [Float] - Eccentricity of the orbit
        f [Float] - True Anomaly [radians]
        mu [Float] - Standard Gravtiational Parameter of the orbit

        Outputs:
        r [Cart Obj] - Position coordinates of the orbit
        v [Cart Obj] - Velocity coordinates of the orbit
        """

        a = self.a
        e = self.e
        f = Angle(f)

        # Helper variables
        n = np.sqrt(mu/a**3)
        E = self.ef2E(f.rad)

        # Position Components
        x = a*(np.cos(E.rad) - e)
        y = a*np.sqrt(1 - e**2)*np.sin(E.rad)
        z = 0

        # Velocity Components
        v_x = -a*n*np.sin(E.rad)/(1 - e*np.cos(E.rad))
        v_y = a*np.sqrt(1 - e**2)*n*np.cos(E.rad)/(1 - e*np.cos(E.rad))
        v_z = 0

        r = Cart(x, y, z)
        v = Cart(v_x, v_y, v_z)

        return r, v

    def rotateOrbitAngles(self, vector, back=False):
        """ Rotates a vector to a heliocentric ecliptic plane

        Inputs:
        vector [Vector3d Obj] - Vector to rotate
        w [Angle Obj] - Argument of perihelion
        i [Angle Obj] - Inclination of the orbit
        O [Angle Obj] - Longitude of the ascending node

        Outputs:
        vector [Vector3d Obj] - Rotated vector
        """

        # Negative Rotation because of rotation matricies
        # Rotation as given in the notes
        if not back:
            vector = vector.rotate(-self.w, "z")
            vector = vector.rotate(-self.i, "x")
            vector = vector.rotate(-self.O, "z")
        else:
            vector = vector.rotate(self.w, "z")
            vector = vector.rotate(self.i, "x")
            vector = vector.rotate(self.O, "z")

        return vector

    def orbit2HeliocentricState(self, mu, f, no_rotate=False, back=False):
        """ Rotate state vectors of position and velocity to heliocentric ecliptic plane

        Inputs:
        k_orbit [KeplerOrbit Obj] - orbital parameters to convert and rotate
        mu [Float] - Standard Gravtiational Parameter of the orbit
        f [Float] - True Anomaly [radians]

        Outputs:
        r [Cart Obj] - Rotated position coordinates of the orbit
        v [Cart Obj] - Rotated velocity coordinates of the orbit
        """

        r, v = self.orbit2State(f, mu)

        if not no_rotate:
            r = self.rotateOrbitAngles(r, back=back)
            v = self.rotateOrbitAngles(v, back=back)

        return r, v

class Observation:
    ''' An observation object with right ascension, declination, jd, and magnitude
    other parameters such as orbit may be added to this object ad hoc from calculations
    for storage

    Inputs:
    ra [float] - Right Ascension of object in degrees
    dec [float] - Declination of object in degrees
    jd [float] - Julian day of observation in days
    mag [float] - Magnitude of observation
    '''

    def __init__(self, ra, dec, jd, mag):

        self.ra = Angle(ra, deg=True)
        self.dec = Angle(dec, deg=True)
        self.jd = jd
        self.mag = mag

    def __str__(self):

        A = "Observation Object \n"
        A +="----------------------- \n"
        A += "RA {:} \n".format(self.ra)
        A += "DEC {:} \n".format(self.dec)
        A += "JD {:} \n".format(self.jd)
        A += "MAG {:} \n".format(self.mag)

        return A


def cart2Radec(cart):

    """
    Extension to Cart object to correctly convert theta and phi to Right Ascension and Declination
    inputs:
    cart [Cart Object] - cart object to convert
    returns:
    ra [Right Ascension Obj] - Right ascension of cart
    dec [float] - Declination of cart
    """
    if not hasattr(cart, "theta") or not hasattr(cart, "phi"):
        print("[WARNING] Cartesian object does not have angles!")
        return None, None

    dec = 90 - cart.theta.deg
    ra  = cart.phi.deg

    return ra, dec