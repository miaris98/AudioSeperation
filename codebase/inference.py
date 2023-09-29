import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
# from sudormrf import  SuDORMRFImprovedNet,SuDORMRFNet
# from asteroid.masknn import SuDORMRF, SuDORMRFImproved,Convtasnet
# from asteroid.utils.torch_utils import script_if_tracing
# from asteroid.models.base_models import BaseEncoderMaskerDecoder replace imports with these in codebase\sudormrf from asteroid
import torch
import torchaudio
from IPython.display import display, Audio
from asteroid.data import LibriMix
from asteroid.models import BaseModel
from speechbrain.pretrained import SepformerSeparation as separator
from torchmetrics import ScaleInvariantSignalDistortionRatio


def main():

    train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
    index = 6
    print(val_set.df.values[index][2])
    print(val_set.df.values[index][3])  # path s1
    print(val_set.df.values[index][4])  # path s2

    # load models
    sepformer = separator.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                       savedir='pretrained_models/sepformer-wsj02mix')
    ConvTasNet = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    DPRNNTasNet = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")
    Sudoimprovednet = BaseModel.from_pretrained('pretrained_models/best_model.pth')
    #ConvTasNet = BaseModel.from_pretrained('pretrained_models/convtasnet/best_model.pth')


    # separate
    filename = (val_set.df.values[index][2])
    filename=filename.split(".")[0]
    filename = filename.split("/")[-1]
    est_sources_sep = sepformer.separate_file(val_set.df.values[index][2])
    torchaudio.save("sepformer1.wav", est_sources_sep[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("sepformer2.wav", est_sources_sep[:, :, 1].detach().cpu(), 8000)
    ConvTasNet.separate(val_set.df.values[index][2], output_dir="ConvTasNet", force_overwrite=True)
    DPRNNTasNet.separate(val_set.df.values[index][2], output_dir="DPRNNTasNet", force_overwrite=True)
    Sudoimprovednet.separate(val_set.df.values[index][2], output_dir="Sudo", resample=True, force_overwrite=True)

    # load predictions
    estse1 = sf.read("sepformer1.wav")[0]
    estse2 = sf.read("sepformer2.wav")[0]

    estc1 = sf.read("ConvTasNet/" + filename+"_est1.wav")[0]
    estc2 = sf.read("ConvTasNet/" + filename+"_est2.wav")[0]

    estd1 = sf.read("DPRNNTasNet/" + filename+"_est1.wav")[0]
    estd2 = sf.read("DPRNNTasNet/" + filename+"_est2.wav")[0]

    ests1 = sf.read("Sudo/" + filename+"_est1.wav")[0]
    ests2 = sf.read("Sudo/" + filename+"_est2.wav")[0]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    show_magspec(ests1, sr=8000, ax=ax[0])
    show_magspec(ests2, sr=8000, ax=ax[1])
    plt.show()

    anechoic_sampled_mixture, _ = torchaudio.load("Sudo/" + filename+"_est1.wav")
    waveform = anechoic_sampled_mixture.detach().numpy()[0]
    plt.plot(waveform)
    plt.show()
    plt.close()
    plt.specgram(waveform)
    display(Audio(waveform, rate=8000))
    plt.show()

    y, sr = librosa.load(val_set.df.values[index][3], sr=8000)
    # Convert the audio data to a tensor
    target = torch.from_numpy(y).float()
    print("metrics for this instance")
    si_sdr = ScaleInvariantSignalDistortionRatio()
    preds = est_sources_sep[:, :, 0].detach().cpu()
    preds = preds.flatten()

    print("si_sdr for sepformer:" + str(si_sdr(torch.as_tensor(preds), target)))
    print("si_sdr for convtasnet:" + str(si_sdr(torch.as_tensor(estc1), target)))
    print("si_sdr for dualpathrnn:" + str(si_sdr(torch.as_tensor(estd2), target)))
    print("si_sdr for sudo:" + str(si_sdr(torch.as_tensor(ests2), target)))


def show_magspec(waveform, **kw):
    return librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(waveform))),
        y_axis="log", x_axis="time",
        **kw
    )


if __name__ == '__main__':
    main()
