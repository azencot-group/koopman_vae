import sys
import os
import numpy as np
from lightning.pytorch import seed_everything

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_cdsvae
from model import KoopmanVAE
from utils.general_utils import imshow_seqeunce, reorder


def define_args():
    # Define the arguments of the model.
    parser = train_cdsvae.define_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')
    parser.add_argument('--model_name', type=str, default=None)

    return parser

if __name__ == '__main__':
    # Define the different arguments and parser them.
    parser = define_args()
    args = parser.parse_args()

    # Set seeds to all the randoms.
    seed_everything(args.seed)

    # Create the model.
    model = KoopmanVAE.load_from_checkpoint(args.model_path)
    model.eval()

    # Load the data.
    data = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch1.npy', allow_pickle=True).item()
    data2 = np.load('/cs/cs_groups/azencot_group/inon/koopman_vae/dataset/batch2.npy', allow_pickle=True).item()
    x = reorder(data['images']).to(model.device)

    # Pass the data through the model.
    z, z_post_koopman, z_post_dropout, Ct, koopman_recon_x, dropout_recon_x = model(
        x)

    # visualize
    index = 0
    titles = ['Original image:', 'Reconstructed image koopman:', 'Original image:', 'Reconstructed image dropout:']
    imshow_seqeunce([[x[index]], [koopman_recon_x[index]], [x[index]], [dropout_recon_x[index]]],
                    titles=np.asarray([titles]).T, figsize=(50, 10), fontsize=50)
