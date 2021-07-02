# FPN
This is the implementation of FPNN used in paper "Fuzzy Prototype Neural Network"

# requirement 
torch=1.7.0
pyyaml=5.3.1
scipy=1.5.4
numpy=1.19.2
scikit-learn=0.23.2


# commond
python main.py --m 'bnn' --d 'wine' --c 'cuda:1' --nl 0.0 --inference 'nuts'
python main.py --m 'bnn' --d 'wine' --c 'cuda:1' --nl 0.5 --inference 'nuts'
python main.py --m 'bnn' --inference 'nuts' --d 'shuttle' --c 'cuda:1' --nl 0.0
