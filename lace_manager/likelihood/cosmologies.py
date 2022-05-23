from lace.cosmo import camb_cosmo

def get_cosmology_from_label(cosmo_label='default'):

    if cosmo_label=='default':
        return camb_cosmo.get_cosmology()
    elif cosmo_label=='low_omch2':
        return camb_cosmo.get_cosmology(omch2=0.11)
    elif cosmo_label=='high_omch2':
        return camb_cosmo.get_cosmology(omch2=0.13)
#    elif cosmo_label=='low_H0':
#        return camb_cosmo.get_cosmology(H0=60)
#    elif cosmo_label=='high_H0':
#        return camb_cosmo.get_cosmology(H0=80)
#    elif cosmo_label=='mnu_03eV':
#        return camb_cosmo.get_cosmology(mnu=0.3)
    else:
        raise ValueError('implement cosmo_label '+cosmo_label)

