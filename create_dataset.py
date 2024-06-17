import os
from csi import *
from scapy.all import *

os.makedirs('data/csv/', exist_ok=True)

save_dir = 'data/csv/'
pcap_dir = '/Users/jeongsehyeon/Downloads/people'

for pcap_fname in os.listdir(pcap_dir):
    df = pcap_to_df(os.path.join(pcap_dir, pcap_fname))
    csv_fname = pcap_fname.split('.')[0] + '.csv'
    df.to_csv(os.path.join(save_dir, csv_fname), index=False)
    print(f'Save {csv_fname}')

# ------ mat ------------
# for mat_fname in os.listdir('data/mat/test_A2/walk/'):
#     df = mat_to_df(os.path.join('data/mat/test_A2/walk/', mat_fname))
#     csv_fname = mat_fname.split('.')[0] + '.csv'
#     df.to_csv(os.path.join(save_dir, csv_fname), index=False)
#     print(f'Save {csv_fname}')
