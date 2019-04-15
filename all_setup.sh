for DIR in lya_cosmo lya_likelihood lya_nuisance p1d_data p1d_emulator post_process setup_simulations user_interface
do
    cd $DIR
    python setup.py install
    cd ..
done	

