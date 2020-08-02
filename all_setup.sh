for DIR in lya_sampler post_process setup_simulations lya_cosmo p1d_data lya_likelihood p1d_emulator user_interface lya_nuisance
do
    cd $DIR
    xargs rm < files_setup.txt
    rm build/lib/*.py
    python3 setup.py install --record files_setup.txt --user
    cd ..
done	

