from lace.cosmo import camb_cosmo

def get_cosmology_from_label(cosmo_label='default'):

    if cosmo_label=='default':
        return camb_cosmo.get_cosmology()
    elif cosmo_label=='low_omch2':
        return camb_cosmo.get_cosmology(omch2=0.11)
    elif cosmo_label=='high_omch2':
        return camb_cosmo.get_cosmology(omch2=0.13)
    elif cosmo_label=='omch2_0115':
        return camb_cosmo.get_cosmology(omch2=0.115)
    elif cosmo_label=='omch2_0125':
        return camb_cosmo.get_cosmology(omch2=0.125)
    elif cosmo_label=='mnu_03':
        return camb_cosmo.get_cosmology(mnu=0.3)
    elif cosmo_label=='mnu_06':
        return camb_cosmo.get_cosmology(mnu=0.6)
    elif cosmo_label=='SHOES':
        return camb_cosmo.get_cosmology(H0=73.0)
    else:
        raise ValueError('implement cosmo_label '+cosmo_label)

