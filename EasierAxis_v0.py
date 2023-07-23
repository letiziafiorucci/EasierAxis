#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: letiziafiorucci & enricoravera
@last_update: 23/07/2023 
"""

#################################################################
# IMPORT SECTION

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

#################################################################
# FUNCTIONS AND CLASSES

class Wigner_coeff():
    """
    This class is a wrapper around the functions to compute angular momentum coupling coefficients
    """

    def threej_symbol(matrix):
        """
        Computes 3j-symbols using Racah's formula.
        Racah, Giulio. Physical Review 62.9-10 (1942): 438.
        ----------
        Parameters:
        - matrix : list of lists
            angular momentum parameters in the form [[a , b, c],[A, B, C]] or [[j1, j2, j3],[m1, m2, m3]]
        ----------
        Returns:
        - result : float
            computed 3j-symbol
        """

        matrix = np.array(matrix)

        # selection rules
        for i in range(3):
            if matrix[1,i]>= -matrix[0,i] and matrix[1,i]<= matrix[0,i]:
                pass
            else:
                return 0

        if np.sum(matrix[1,:])==0:
            pass
        else:
            return 0

        if matrix[0,-1] >= np.abs(matrix[0,0]-matrix[0,1]) and matrix[0,-1] <= matrix[0,0]+matrix[0,1]:
            pass
        else:
            return 0

        if np.sum(matrix[0,:])%1==0:
            if matrix[1,0] == matrix[1,1] and matrix[1,1] == matrix[1,2]:
                if np.sum(matrix[0,:])%2==0:
                    pass
                else:
                    return 0
            else:
                pass
        else:
            return 0

        # coefficient calculation
        a = matrix[0,0]
        b = matrix[0,1]
        c = matrix[0,2]
        A = matrix[1,0]
        B = matrix[1,1]
        C = matrix[1,2]

        n_min_list = [0, -c+b-A, -c+a+B]
        n_min = max(n_min_list)
        n_max_list = [a+b-c, b+B, a-A]
        n_max = min(n_max_list)
        if a-b-C <0:
            factor = (1/(-1))**np.abs(a-b-C)
        else:
            factor = (-1)**(a-b-C)
        sect1 = factor*(fact(a+b-c)*fact(b+c-a)*fact(c+a-b)/fact(a+b+c+1))**(0.5)
        sect2 = (fact(a+A)*fact(a-A)*fact(b+B)*fact(b-B)*fact(c+C)*fact(c-C))**(0.5)
        sect3 = 0
        for n in np.arange(n_min,n_max+1,1):
            sect3 += (-1)**n*(fact(n)*fact(c-b+n+A)*fact(c-a+n-B)*fact(a+b-c-n)*fact(a-n-A)*fact(b-n+B))**(-1)

        result = sect1*sect2*sect3

        return result

    def sixj_symbol(matrix):
        """
        Computes 6j-symbols using Racah's formula.
        Racah, Giulio. Physical Review 62.9-10 (1942): 438.
        ----------
        Parameters:
        - matrix : list of lists
            angular momentum parameters in the form [[a, b, c],[A, B, C]] or [[j1, j2, j3],[j4, j5, j6]]
        ----------
        Returns:
        - result : float
            computed 6j-symbol
        """

        matrix = np.array(matrix)

        # selection rules
        triads = np.array([[matrix[0,0], matrix[0,1], matrix[0,2]], [matrix[0,0], matrix[1,1], matrix[1,2]],
                  [matrix[1,0], matrix[0,1], matrix[1,2]], [matrix[1,0], matrix[1,1], matrix[0,2]]])

        for i in range(len(triads[:,0])):
            if np.sum(matrix[0,:])%1==0:
                pass
            else:
                return 0

            if triads[i,2] >= np.abs(triads[i,0]-triads[i,1]) and triads[i,2] <= triads[i,0]+triads[i,1]:
                pass
            else:
                return 0


        def f(aa, bb, cc):
            return (fact(aa+bb-cc)*fact(aa-bb+cc)*fact(-aa+bb+cc)/fact(aa+bb+cc+1))**(0.5)

        # coefficient calculation
        a = matrix[0,0]
        b = matrix[0,1]
        c = matrix[0,2]
        A = matrix[1,0]
        B = matrix[1,1]
        C = matrix[1,2]

        n_min_list = [0, a+A-c-C, b+B-c-C]
        n_min = max(n_min_list)
        n_max_list = [a+b+A+B+1, a+b-c, A+B-c, a+B-C, A+b-C]
        n_max = min(n_max_list)

        sect1 = (-1)**(a+b+A+B)*f(a,b,c)*f(A,B,c)*f(A,b,C)*f(a,B,C)

        sect2 = 0
        for n in np.arange(n_min,n_max+1,1):
            sect2 += (-1)**n*fact(a+b+A+B+1-n)/(fact(n)*fact(a+b-c-n)*fact(A+B-c-n)*fact(a+B-C-n)*fact(A+b-C-n)*fact(-a-A+c+C+n)*fact(-b-B+c+C+n))

        result = sect1*sect2

        return result

def fact(number):
    """
    Computes the factorial of number
    ----------
    Parameters:
    - number : int
    ----------
    Returns:
    - factorial : int
    """
    number = int(number)
    if number < 0:
        print('negative number in factorial')
        exit()
    else:
        factorial = 1
        for i in range(1,number+1,1):
            factorial *= i
        return int(factorial)

def princ_comp_sort(w, v=np.zeros((3,3))):
    """
    Sorts the eigenvalues of a matrix in descending order, then sorts the associated eigenvectors accordingly
    ----------
    Parameters:
    - w : 1darray
        matrix eigenvalues
    - v : 2darray
        matrix eigenvectors column-wise
    ----------
    Returns:
    - wst : 1darray
        sorted eigenvalues
    - vst : 2darray
        sorted eigenvectors column-wise
    """

    indices = np.argsort(w)[::-1]
    wst = w[indices]
    vst = v[:,indices]
    return wst, vst

def from_car_to_sph(coord):
    """
    Converts a set of coordinates from cartesian to spherical.
    They must be centered in [0,0,0] in order to work properly.
    ----------
    Parameters:
    - coord : 2darray
        set of cartesian coordinates
    ----------
    Returns:
    - coord_conv : 2darray
        set of converted spherical coordinates
    """
    coord_conv = np.zeros_like(coord)
    for i in range(len(coord[:,0])):
        coord_conv[i,0] = np.linalg.norm(coord[i,:])
        coord_conv[i,1] = np.arccos(coord[i,2]/coord_conv[i,0])
        coord_conv[i,2] = np.arctan2(coord[i,1],coord[i,0])

    return coord_conv

def from_sph_to_car(coord_sph):
    """
    Converts a set of coordinates from spherical to cartesian.
    The angles must be expressed in radians.
    ----------
    Parameters:
    - coord_sph : 2darray
        set of spherical coordinates
    ----------
    Returns:
    - coord_car : 2darray
        set of converted cartesian coordinates
    """
    coord_car = np.zeros_like(coord_sph)
    for i in range(len(coord_car[:,0])):
        coord_car[i,0] = coord_sph[i,0]*np.cos(coord_sph[i,2])*np.sin(coord_sph[i,1])
        coord_car[i,1] = coord_sph[i,0]*np.sin(coord_sph[i,2])*np.sin(coord_sph[i,1])
        coord_car[i,2] = coord_sph[i,0]*np.cos(coord_sph[i,1])

    return coord_car

elem_cpk = {  # color codes for elements
        # Basics
        'H' : 'lightgray',
        'C' : 'k',
        'N' : 'b',
        'O' : 'r',
        # Halogens
        'F' : 'tab:green',
        'Cl': 'g',
        'Br': 'maroon',
        'I' : 'darkviolet',
        # Noble gases
        'He': 'c',
        'Ne': 'c',
        'Ar': 'c',
        'Kr': 'c',
        'Xe': 'c',
        # Common nonmetals
        'P' : 'orange',
        'S' : 'y',
        'B' : 'tan',
        # Metals
        #   Alkali
        'Li': 'violet',
        'Na': 'violet',
        'K' : 'violet',
        'Rb': 'violet',
        'Cs': 'violet',
        #   Alkali-earth
        'Be': 'darkgreen',
        'Mg': 'darkgreen',
        'Ca': 'darkgreen',
        'Sr': 'darkgreen',
        'Ba': 'darkgreen',
        #   Transition, I series
        'Sc': 'steelblue',
        'Ti': 'steelblue',
        'V' : 'steelblue',
        'Cr': 'steelblue',
        'Mn': 'steelblue',
        'Fe': 'steelblue',
        'Co': 'steelblue',
        'Ni': 'steelblue',
        'Cu': 'steelblue',
        'Zn': 'steelblue',
        #   Transition, II series
        'Y' : 'deepskyblue',
        'Zr': 'deepskyblue',
        'Nb': 'deepskyblue',
        'Mo': 'deepskyblue',
        'Tc': 'deepskyblue',
        'Ru': 'deepskyblue',
        'Rh': 'deepskyblue',
        'Pd': 'deepskyblue',
        'Ag': 'deepskyblue',
        'Cd': 'deepskyblue',
        #   Transition, III series
        'La': 'cadetblue',
        'Hf': 'cadetblue',
        'Ta': 'cadetblue',
        'W' : 'cadetblue',
        'Re': 'cadetblue',
        'Os': 'cadetblue',
        'Ir': 'cadetblue',
        'Pt': 'cadetblue',
        'Au': 'cadetblue',
        'Hg': 'cadetblue',
        #   Lanthanides
        'Ce': 'teal',
        'Pr': 'teal',
        'Nd': 'teal',
        'Pm': 'teal',
        'Sm': 'teal',
        'Eu': 'teal',
        'Gd': 'teal',
        'Tb': 'teal',
        'Dy': 'teal',
        'Ho': 'teal',
        'Er': 'teal',
        'Tm': 'teal',
        'Yb': 'teal',
        'Lu': 'teal',
        # Default color for all the others
        '_' : 'tab:pink',
        }

def ground_term_legend(conf):
    """
    Returns the ground state for a given fn configuration.
    ----------
    Parameters:
    - conf : str
        fn configuration e.g. 'f3'
    ----------
    Returns:
    - legenda[conf] : str
        ground state term
    """
    legenda = {'f1':'2F (5/2)',
               'f2':'3H (4)',
               'f3':'4I (9/2)',
               'f4':'5I (4)',
               'f5':'6H (5/2)',
               'f8':'7F (6)',
               'f9':'6H (15/2)',
               'f10':'5I (8)',
               'f11':'4I (15/2)',
               'f12':'3H (6)',
               'f13':'2F (7/2)'}
    return legenda[conf]

def state_legend(L):
    """
    Converts the term symbol in the corresponding L value if L is a str, or viceversa if L is a number.
    ----------
    Parameters:
    - L : str or int
        L symbol
    ----------
    Returns:
    - ret_val : int or str
        L symbol
    """
    legenda = {'S':0,
               'P':1,
               'D':2,
               'F':3,
               'G':4,
               'H':5,
               'I':6,
               'K':7,
               'L':8,
               'M':9,
               'N':10,
               'O':11,
               'Q':12,
               'R':13,
               'T':14,
               'U':15,
               'V':16}

    if isinstance(L, str):
        ret_val = legenda[L]
    else:
        inv_map = {str(v): k for k, v in legenda.items()}
        ret_val = inv_map[L]
    return ret_val

def simpre_legend(conf):
    """
    It is used to assess which SIMPRE index corresponds to a given configuration.
    ----------
    Parameters:
    - conf : str
        fn configuration e.g. 'f3'
    ----------
    Returns:
    - legenda[conf] : int
        SIMPRE lanthanoid ion index
    """
    legenda = {'f1':1,
               'f2':2,
               'f3':3,
               'f4':4,
               'f5':5,
               'f8':6,
               'f9':7,
               'f10':8,
               'f11':9,
               'f12':10,
               'f13':11}
    return legenda[conf]

def w_inp_dat(coord_tot, charges_tot, num):
    """
    Creates a file named 'simpre.dat', which contains coordinates and charges, to be used as input for a SIMPRE calculation.
    As such, only the negative charges are considered.
    ----------
    Parameters:
    - coord_tot : 2darray
        cartesian coordinates of the ligands
    - charges_tot : sequence
        ligand charges (with their sign)
    - num : int
        SIMPRE index for a configuration
    """

    coord = []
    charges = []
    for i in range(coord_tot.shape[0]):
        if charges_tot[i]<=0:
            coord.append(coord_tot[i,:])
            charges.append(np.abs(charges_tot[i]))
    coord = np.array(coord)
    if coord.size==0:
        print('WARNING: Only positive charged ligands found. The "simpre.dat" file can not be generated.')
        return 0

    with open('simpre.dat', 'w') as f:
        f.write('1=Ce3+, 2=Pr3+, 3=Nd3+, 4=Pm3+, 5=Sm3+, 6=Tb3+, 7=Dy3+, 8=Ho3+, 9=Er3+, 10=Tm3+, 11=Yb3+, 12=user\n')
        f.write('\n')
        f.write(' '+f'{int(num):2.0f}'+'    !ion code (from 1 to 12, see above)\n')
        f.write('  '+f'{coord.shape[0]}'+'    !number of effective charges\n')
        for i in range(len(coord[:,0])):
            f.write('{:-3}'.format(i+1))
            for j in range(3):
                f.write('{:13.7f}'.format(coord[i,j]))
            f.write('{:13.7f}'.format(charges[i])+'\n')
        f.write('\n')
        f.write('\n')
        f.close()

def w_bild(axis):
    """
    Creates a file named 'EasyAxis.bild', which is used to instruct molecular visualization programs (e.g. chimera) to draw the easy axis as a red cylinder.
    ----------
    Parameters:
    - axis : sequence
        easy axis cartesian coordinates [x,y,z]
    """
    with open('EasyAxis.bild', 'w') as f:
        f.write('.color red\n')
        f.write('.cylinder '+f'{-axis[0]:.7f}'+' '+f'{-axis[1]:.7f}'+' '+f'{-axis[2]:.7f}'+' '+f'{axis[0]:.7f}'+' '+f'{axis[1]:.7f}'+' '+f'{axis[2]:.7f}'+' '+'0.1')
        f.close()

def coeff_multipole_moments(conf, J, M, L=0, S=0):
    """
    Computes the multipole expansion coefficients (Ak = sqrt(4pi/(2k+1))·ck) of a 4f electron-density distribution for a given term of a fn configuration.
    The equations used are eq. 10 and 12 from Ch. II p. 290 of Sievers, J. Zeitschrift für Physik B Condensed Matter 45.4 (1982): 289-296.
    ----------
    Parameters:
    - conf : str
        fn configuration e.g. 'f3'
    - J : float
        total angular momentum
    - M : float
        magnetic angular momentum
    - L : float
        orbital angular momentum
    - S : float
        spin angular momentum
    ----------
    Returns:
    - coeff : list
        list of Ak coefficients
    """

    coeff = []
    for k in range(2,6+1,2):
        if int(conf[1:])>7:
            pref = (-1)**(J-M)*7/np.sqrt(4*np.pi)*Wigner_coeff.threej_symbol([[J,k,J],[-M,0,M]])/Wigner_coeff.threej_symbol([[J,k,J],[-J,0,J]])
            pref2 = np.sqrt(2*k+1)*Wigner_coeff.threej_symbol([[k,3,3],[0,0,0]])
            delta = 0
            somma = 0
            for i in range(1,int(conf[1:])-7+1):
                somma += (-1)**i*Wigner_coeff.threej_symbol([[k,3,3],[0,4-i,i-4]])
            value = pref*(somma*pref2+delta)*np.sqrt(4*np.pi/(2*k+1))
            coeff.append(value)
        else:
            pref = (-1)**(2*J-M+L+S)*7/np.sqrt(4*np.pi)*(2*J+1)*np.sqrt(2*k+1)*Wigner_coeff.threej_symbol([[J,k,J],[-M,0,M]])/Wigner_coeff.threej_symbol([[L,k,L],[-L,0,L]])
            pref2 = Wigner_coeff.sixj_symbol([[L,J,S],[J,L,k]])*Wigner_coeff.threej_symbol([[k,3,3],[0,0,0]])
            somma = 0
            for i in range(1,int(conf[1:])+1):
                somma += (-1)**i*Wigner_coeff.threej_symbol([[k,3,3],[0,4-i,i-4]])
            value = pref*pref2*somma*np.sqrt(4*np.pi/(2*k+1))
            coeff.append(value)

    return coeff

def r_expect(k, conf):
    """
    Returns the average radial integral of a rare earth ion (<r^k>) for a given k and fn configuration.
    Tables from S. Edvarsson, M. Klintenberg, Journal of Alloys and Compounds. 1998, 275–277, 230.
    ----------
    Parameters:
    - k : int
        spherical harmonics rank
    - conf : str
        fn configuration e.g. 'f3'
    ----------
    Returns:
    - legenda[k][conf] : float
        <r^k> value for the given rank and configuration
    """
    legenda = {'2':{'f1':1.456, 'f2':1.327, 'f3':1.222, 'f4':1.135, 'f5':1.061, 'f6':0.997, 'f7':0.942, 'f8':0.893, 'f9':0.849, 'f10':0.810, 'f11':0.773, 'f12':0.740, 'f13':0.710},
               '4':{'f1':5.437, 'f2':4.537, 'f3':3.875, 'f4':3.366, 'f5':2.964, 'f6':2.638, 'f7':2.381, 'f8':2.163, 'f9':1.977, 'f10':1.816, 'f11':1.677, 'f12':1.555, 'f13':1.448},
               '6':{'f1':42.26, 'f2':32.65, 'f3':26.12, 'f4':21.46, 'f5':17.99, 'f6':15.34, 'f7':13.36, 'f8':11.75, 'f9':10.44, 'f10':9.345, 'f11':8.431, 'f12':7.659, 'f13':7.003}}
    return legenda[k][conf]

def freeion_charge_dist(theta, phi, A2, A4, A6, bin=1e-10):
    """
    Computation of a given point on the Sievers surface.
    The equation used is eq. 20 from Ch. III p 292 of Sievers, J. Zeitschrift für Physik B Condensed Matter 45.4 (1982): 289-296.
    ----------
    Parameters:
    - theta : float
        polar angle in radians
    - phi : float
        azimuthal angle in radians
    - A2 : float
        rank-2 coefficient of multipolar expansion
    - A4 : float
        rank-4 coefficient of multipolar expansion
    - A6 : float
        rank-6 coefficient of multipolar expansion
    - bin : float
        threshold below which the returned value is zero
    ----------
    Returns:
    - coeff : list
        list of Ak coefficients
    """

    c2 = A2/np.sqrt(4*np.pi/(2*2+1))
    c4 = A4/np.sqrt(4*np.pi/(2*4+1))
    c6 = A6/np.sqrt(4*np.pi/(2*6+1))

    val = 3/(4*np.pi) + c2*scipy.special.sph_harm(0,2,phi,theta).real + c4*scipy.special.sph_harm(0,4,phi,theta).real + c6*scipy.special.sph_harm(0,6,phi,theta).real
    if np.abs(val)<bin:
        val=0
    ret_val = (val)**(1/3)
    return ret_val

def plot_sievers4efg(conf, J, M, L, S, w, v, data=None):
    """
    Given a fn configuration and M, computes and plots the correspondent Sievers surface on the point charge model rotated in the reference frame of V.
    The eigenvectors of V are also displayed as red arrows in their reference system.
    The point charge model itself is displayed only if a suitable data matrix is provided.
    ----------
    Parameters:
    - conf : str
        fn configuration e.g. 'f3'
    - J : float
        total angular momentum
    - M : float
        magnetic angular momentum
    - L : float
        orbital angular momentum
    - S : float
        spin angular momentum
    - w : sequence
        eigenvalues of V
    - v : 2darray
        eigenvectors of V column-wise
    - data : 2darray
        array of shape (n,5) where n is the number of ligands.
        Each row consists of: [label, x, y, z, charge]
    """

    A2, A4, A6 = coeff_multipole_moments(conf, J, M, L, S)
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            r[i,j] = freeion_charge_dist(theta[i,j], phi[i,j], A2, A4, A6)*1.5
    xyz = np.array([r*np.sin(theta) * np.sin(phi),
                    r*np.sin(theta) * np.cos(phi),
                    r*np.cos(theta)])
    X, Y, Z = xyz
    massimo = np.amax(xyz)

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.set_xlabel('x',fontsize=13)
    ax.set_ylabel('y',fontsize=13)
    ax.set_zlabel('z',fontsize=13)
    asse = 3
    ax.set_xlim(-asse, asse)
    ax.set_ylim(-asse, asse)
    ax.set_zlim(-asse, asse)
    ax.plot_surface(X, Y, Z, edgecolor='b', lw=0.25, rstride=2, cstride=2,
                alpha=0.3)
    if data is not None:
        for i in range(data.shape[0]):
            vector = data[i,1:-1]
            ax.plot([0.,vector[0]],[0.,vector[1]],[0.,vector[2]],'--',lw=1,c='k')
            if data[i,0] in elem_cpk.keys():
                ax.scatter(vector[0],vector[1],vector[2],'o',c = elem_cpk[data[i,0]],lw=6)
            else:
                ax.scatter(vector[0],vector[1],vector[2],'o',c = elem_cpk['_'],lw=6)

    ax_label = ['x','y','z']
    for i in range(3):
        vector = v[:,i]*2.5
        ax.quiver(0.,0.,0.,vector[0],vector[1],vector[2],color='r', arrow_length_ratio=0.1)
        ax.text(vector[0]+0.1*np.sign(vector[0]),vector[1]+0.1*np.sign(vector[1]),vector[2]+0.1*np.sign(vector[2]),ax_label[i], size=16)#f'{w[i]:.4e}')
    ax.set_title('conf = '+conf+'  J = '+str(J)+'  M = '+str(M))

    plt.show()

def figure_justcoord(data):
    """
    Plots the point charge model.
    ----------
    Parameters:
    - data : 2darray
        array of shape (n,5) where n is the number of ligands.
        Each row consists of: [label, x, y, z, charge]
    """

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.set_xlabel('x',fontsize=13)
    ax.set_ylabel('y',fontsize=13)
    ax.set_zlabel('z',fontsize=13)
    asse = 2.5
    ax.set_xlim(-asse, asse)
    ax.set_ylim(-asse, asse)
    ax.set_zlim(-asse, asse)

    if data is not None:
        for i in range(data.shape[0]):
            vector = data[i,1:-1]
            ax.plot([0.,vector[0]],[0.,vector[1]],[0.,vector[2]],'--',lw=1,c='k')
            if data[i,0] in elem_cpk.keys():
                ax.scatter(vector[0],vector[1],vector[2],'o',c = elem_cpk[data[i,0]],lw=8)
            else:
                ax.scatter(vector[0],vector[1],vector[2],'o',c = elem_cpk['_'],lw=8)
            ax.text(vector[0]+0.4*np.sign(vector[0]),vector[1]+0.4*np.sign(vector[1]),vector[2]+0.4*np.sign(vector[2]),data[i,-1], size=20)

    plt.show()

def multipole_exp_tensor(conf, J, M, L, S, data):
    """
    Computes the lowest component of the multipole expansion tensor according to equation 1 of the manual.
    ----------
    Parameters:
    - conf : str
        fn configuration e.g. 'f3'
    - J : float
        total angular momentum
    - M : float
        magnetic angular momentum
    - L : float
        orbital angular momentum
    - S : float
        spin angular momentum
    - data : 2darray
        array of shape (n,5) where n is the number of ligands.
        Each row consists of: [label, x, y, z, charge]
    ----------
    Returns:
    - eiv : float
        component of multipole expansion tensor
    - A : list
        list of Ak Sievers coefficients
    """

    A2, A4, A6 = coeff_multipole_moments(conf, J, M, L, S)
    A = [A2, A4, A6]
    coord = data[:,1:-1]
    coord_sph = from_car_to_sph(coord)
    eiv = 0
    for i in range(coord.shape[0]):
        eiv += -data[i,-1]*A2*r_expect(str(2),conf)*(3*coord[i,2]**2-coord_sph[i,0]**2)/coord_sph[i,0]**5*1/4*np.sqrt(5/np.pi)*4*np.pi/5
        eiv += -data[i,-1]*A4*r_expect(str(4),conf)*(35*coord[i,2]**4-30*coord[i,2]**2*coord_sph[i,0]**2+3*coord_sph[i,0]**4)/coord_sph[i,0]**9*3/16*np.sqrt(1/np.pi)*4*np.pi/9
        eiv += -data[i,-1]*A6*r_expect(str(6),conf)*(231*coord[i,2]**6-315*coord[i,2]**4*coord_sph[i,0]**2+105*coord[i,2]**2*coord_sph[i,0]**4-5*coord_sph[i,0]**6)/coord_sph[i,0]**13*1/32*np.sqrt(13/np.pi)*4*np.pi/13

    return eiv, A

def EFG_tensor(data):
    """
    Computes the electric field gradient tensor V according to equation 2 in the manual.
    ----------
    Parameters:
    - data : 2darray
        array of shape (n,5) where n is the number of ligands.
        Each row consists of: [label, x, y, z, charge]
    ----------
    Returns:
    - V : 2darray
        electric field gradient tensor
    """

    V_tens = np.zeros((3,3))
    for i in range(data.shape[0]):
        V = np.zeros((3,3))
        V[0,0] = 2*data[i,1]**2-data[i,2]**2-data[i,3]**2
        V[1,1] = 2*data[i,2]**2-data[i,1]**2-data[i,3]**2
        V[2,2] = 2*data[i,3]**2-data[i,1]**2-data[i,2]**2
        V[1,0] = 3*data[i,1]*data[i,2]
        V[0,1] = V[1,0]
        V[2,0] = 3*data[i,3]*data[i,1]
        V[0,2] = V[2,0]
        V[2,1] = 3*data[i,3]*data[i,2]
        V[1,2] = V[2,1]
        V_tens += -data[i,-1]* (1/np.sqrt((data[i,1]**2+data[i,2]**2+data[i,3]**2)**5))*V

    return V_tens

def consecutive_eq_el(arr):
    """
    Find consecutive equal elements in an array and return their indices in a list of lists.
    ----------
    Parameters:
    - arr : list
        The input list containing elements to be checked for consecutive equality.
    ----------
    Returns:
    - equal_elements_indices : list of lists
        A list of lists where each inner list contains the indices of consecutive occurrences of the same element. If no consecutive equal elements are found, an empty list is returned.
    """

    prev_element = 1e10  #just a very unlikely value (can be changed if needed)
    sequence_start_idx = 1e10
    equal_elements_indices = []

    for idx, element in enumerate(arr):
        if round(element,10) != round(prev_element,10):
            if sequence_start_idx is not None and idx - sequence_start_idx >= 2:
                equal_elements_indices.append(list(range(sequence_start_idx, idx)))

            sequence_start_idx = idx

        prev_element = element

    if sequence_start_idx is not None and len(arr) - sequence_start_idx >= 2:
        equal_elements_indices.append(list(range(sequence_start_idx, len(arr))))

    return equal_elements_indices

#################################################################
# MAIN PROGRAM

def main(conf, coord, charges, labels, filename):
    """
    Main program.
    ----------
    Parameters:
    - conf : str
        fn configuration e.g. 'f3'
    - coord : 2darray
        set of cartesian coordinates of ligands
    - charges : sequence
        ligand charges (with their sign)
    - labels : sequence
        ligand labels
    - filename : str
        input file name
    """

    ground = ground_term_legend(conf)
    S = (int(ground[0])-1)/2
    L = state_legend(ground[1])
    splitg = ground.split('(')
    J = eval(splitg[-1][:-1])

    # "data" preparation [label, x, y, z, charges (with sign)]
    data = np.zeros((coord.shape[0],5), dtype='object')
    data[:,1:-1] = coord
    data[:,-1] = np.array(charges)
    data[:,0] = labels
    V_tens = EFG_tensor(data)
    #check if V_tens elements are all 0
    if np.all(np.round(V_tens, 10)==0):
        print('ERROR: All zero elements detected in V tensor.')
        exit()

    # eigenvalues and eigenvectors calculation
    w,v = np.linalg.eigh(V_tens)

    #choice of the dominant component in rank-2
    #this allows to understand whether the ground state is oblate or prolate
    c21 = coeff_multipole_moments(conf, J, J, L, S)[0]
    opt1 = c21*w
    c22 = coeff_multipole_moments(conf, J, min(np.arange(J,-0.5,-1)), L, S)[0]
    opt2 = c22*w

    if min(opt1) < min(opt2):
        w, v = princ_comp_sort(opt1, v)
        if c21>=0:
            quad_approx = 'PROLATE'
        else:
            quad_approx = 'OBLATE'
    else:
        w, v = princ_comp_sort(opt2, v)
        if c22>=0:
            quad_approx = 'PROLATE'
        else:
            quad_approx = 'OBLATE'
    v_opt = np.copy(v)

    #check for degenerate eigenvalues
    indices_eq = consecutive_eq_el(w)
    labels = ['x','y','z']
    comment = None
    if indices_eq:
        for i in range(len(indices_eq)):
            lindices_eq = [labels[i] for i in indices_eq[i]]
            comment = 'Any combination of eigenvectors '+', '.join(lindices_eq)+' is equally valid\n'

    # coordinates rotation in the CF reference system
    rot_coord = np.zeros_like(coord)
    for ii in range(coord.shape[0]):
        rot_coord[ii,:] = v.T@coord[ii,:]
    v = v.T@v
    data[:,1:-1] = rot_coord

    # calculation of the other terms in multipole expansion for ground state prediction
    sph_coord = from_car_to_sph(rot_coord)
    res = np.zeros(len(np.arange(J,-0.5,-1)))
    c_tab = []
    for jj,M in enumerate(np.arange(J,-0.5,-1)):
        res[jj], c = multipole_exp_tensor(conf, J, M, L, S, data)
        c_tab.append(c)

    M_low = np.arange(J,-0.5,-1)[res.argmin()]

    #plot of the Sievers surface in the Point Charge model framework
    #To change the axes length modify the variable 'asse' in the plot_sievers4efg function
    plot_sievers4efg(conf, J, M_low, L, S, w, v, data)

    #generation of the simpre.dat input file
    #(remember that in SIMPRE only positive charge values are allowed)
    w_inp_dat(rot_coord, charges, simpre_legend(conf))
    w_bild(v_opt[:,-1]*5)  #*5 just to make it more visible

    #output generation
    with open('output.out', 'w') as f:
        f.write('************************************************************\n')
        f.write('**                      ============                      **\n')
        f.write('**                       EasierAxis                       **\n')
        f.write('**                      ============                      **\n')
        f.write('**                                                        **\n')
        f.write('** Authors: LETIZIA FIORUCCI, ENRICO RAVERA               **\n')
        f.write('** email: fiorucci@cerm.unifi.it, ravera@cerm.unifi.it    **\n')
        f.write('** CERM and Department of Chemistry “Ugo Schiff”,         **\n')
        f.write('** University of Florence (Italy)                         **\n')
        f.write('************************************************************\n')
        f.write('\n')
        f.write('\n')
        f.write('Configuration: '+conf+'\n')
        f.write('(L='+str(L)+', S='+str(S)+', J='+str(J)+')\n')
        f.write('\n')
        f.write('Input filename: '+filename+'\n')
        f.write('label\tcharge\tr_disp\t\tx\t\ty\t\tz\n')
        for i,line in enumerate(file):
            f.write(data[i,0]+'\t'+f'{data[i,-1]:.4f}'+'\t'+f'{rdisp[i]:.7f}'+'\t'+f'{coord[i,0]:.7f}'+'\t'+f'{coord[i,1]:.7f}'+'\t'+f'{coord[i,2]:.7f}'+'\n')
        for values in charges:
            if values>0:
                f.write('\n### WARNING: The input contains positive charges but in "simpre.dat" only negative ligands will be reported. ###\n')
                break
        f.write('\n'+'-'*80+'\n\n')
        f.write('Multipole moments:\n')
        f.write('M\tA2\t\tA4\t\tA6\n')
        for i,M in enumerate(np.arange(J,-0.5,-1)):
            f.write(str(M)+'\t'+f'{c_tab[i][0]:.6f}'+'\t'+f'{c_tab[i][1]:.6f}'+'\t'+f'{c_tab[i][2]:.6f}'+'\n')
        f.write('\n')
        f.write('V tensor:\n')
        for i in range(V_tens.shape[0]):
            f.write(f'{V_tens[i,0]:.6f}'+'\t'+f'{V_tens[i,1]:.6f}'+'\t'+f'{V_tens[i,2]:.6f}'+'\n')
        f.write('\n')
        f.write('Eigenvalues * A2 (M='+str(max(np.arange(J,-0.5,-1)))+'): '+str(opt1)+'\n')
        f.write('Eigenvalues * A2 (M='+str(min(np.arange(J,-0.5,-1)))+'): '+str(opt2)+'\n')
        f.write('\n')
        f.write('Ordered Eigenvalues * A2: '+str(w)+'\n')
        f.write('Ordered Eigenvectors: (molecular frame)\n')
        f.write('x\t\ty\t\tz\n')
        for i in range(v_opt.shape[0]):
            f.write(f'{v_opt[i,0]:.6f}'+'\t'+f'{v_opt[i,1]:.6f}'+'\t'+f'{v_opt[i,2]:.6f}'+'\n')
        if comment:
            f.write(comment)
        f.write('\n'+'-'*80+'\n\n')
        f.write('Quadrupolar approximation: '+quad_approx+'\n')
        f.write('Lower M: '+str(M_low)+'\n')
        f.write('Rotated coordinates:\n')
        f.write('label\tx\t\ty\t\tz\n')
        for i in range(rot_coord.shape[0]):
            f.write(data[i,0]+'\t'+f'{rot_coord[i,0]:.7f}'+'\t'+f'{rot_coord[i,1]:.7f}'+'\t'+f'{rot_coord[i,2]:.7f}'+'\n')
        f.write('\n************************************************************\n')
        f.close()

if __name__ == '__main__':

    #just reading the input file and preparing the variables for the main function
    filename = sys.argv[1]
    file = open(filename).readlines()
    coord = []
    charges = []
    labels = []
    rdisp = []
    for i,line in enumerate(file):
        splitline = line.split('\t')
        labels.append(splitline[0])
        charges.append(eval(splitline[1]))
        rdisp.append(float(splitline[2]))
        coord.append([float(splitline[j]) for j in range(3,len(splitline))])
    coord = np.array(coord)
    rdisp = np.array(rdisp)
    sph_coord = from_car_to_sph(coord)
    sph_coord[:,0] -= rdisp
    coord = from_sph_to_car(sph_coord)
    charges = np.array(charges)

    conf = sys.argv[2]
    try:
        simpre_legend(conf)
    except:
        print('ERROR: The selected configuration is not valid. Try with one of the following:\nf1 f2 f3 f4 f5 f8 f9 f10 f11 f12 f13\n')
        exit()

    main(conf, coord, charges, labels, filename)

##################################################################


#
