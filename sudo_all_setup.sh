for DIR in lya_sampler post_process setup_simulations lya_cosmo p1d_data lya_likelihood p1d_emulator user_interface lya_nuisance
do
    cd $DIR
    sudo xargs rm < files_setup.txt
    sudo rm build/lib/*.py
    sudo python3 setup.py install --record files_setup.txt
    cd ..
done	

