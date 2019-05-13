#for DIR in lya_sampler post_process setup_simulations lya_cosmo p1d_data lya_likelihood p1d_emulator user_interface lya_nuisance
for DIR in post_process setup_simulations lya_cosmo p1d_data lya_likelihood p1d_emulator user_interface lya_nuisance
do
    cd $DIR
    python setup.py install
    cd ..
done	

