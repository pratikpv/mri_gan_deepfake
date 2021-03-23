import torch
import torch.nn as nn
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

log_dir_t03 = 'logs/23-Feb-2021_21_22_21'  # tau=0.3
log_dir_t05 = 'logs/02-Mar-2021_12_47_10'  # tau=0.5
#log_dir_t07 = 'logs/04-Mar-2021_09_12_48'  # tau=0.7 org. but losses file corrupted
log_dir_t07 = 'logs/15-Mar-2021_08_38_45'  # tau=0.7 new

log_dir_t10 = 'logs/13-Mar-2021_09_57_29'  # tau=1
log_dir_t00 = 'logs/09-Mar-2021_12_48_48'  # tau=0 Fix

xylabel_fontsize = 22
tick_fontsize = 20
legend_fontsize = 20

################################
# t = 0.0
################################
ssim_report_file_t00 = os.path.join(log_dir_t00, 'MRI_GAN', 'ssim_report_file.pkl')
losses_file_t00 = os.path.join(log_dir_t00, 'MRI_GAN', 'losses.pkl')
metadata_file_t00 = os.path.join(log_dir_t00, 'MRI_GAN', 'mri_metadata.pkl')
losses_t00 = pickle.load(open(losses_file_t00, "rb"))
ssim_report_t00 = pickle.load(open(ssim_report_file_t00, "rb"))
metadata_t00 = pickle.load(open(metadata_file_t00, "rb"))
if metadata_t00['model_params']['tau'] == 0.0:
    print('metadata_t00 is valid')
else:
    #raise Exception('Bad tau')
    pass

df_ssim_t00 = pd.DataFrame(data=ssim_report_t00,
                           columns=['epoch', 'local_batch_num', 'global_batch_num', 'mean_ssim_t0.0'])
df_ssim_t00 = df_ssim_t00[['global_batch_num', 'mean_ssim_t0.0']].set_index('global_batch_num')

df_losses_t00 = pd.DataFrame(data=losses_t00, columns=['epoch', 'local_batch_num', 'global_batch_num',
                                                       'loss_G_t0.0', 'loss_GAN_t0.0', 'loss_pixel_t0.0',
                                                       'loss_ssim_t0.0',
                                                       'loss_D_t0.0', 'loss_real_t0.0', 'loss_fake_t0.0'])
df_losses_t00 = df_losses_t00.drop(labels=['epoch', 'local_batch_num'], axis=1)  # .set_index('global_batch_num')
df_G_only_t00 = df_losses_t00[['global_batch_num', 'loss_G_t0.0']].set_index('global_batch_num')
df_D_only_t00 = df_losses_t00[['global_batch_num', 'loss_D_t0.0']].set_index('global_batch_num')
df_pixel_loss_t00 = df_losses_t00[['global_batch_num', 'loss_pixel_t0.0']].set_index('global_batch_num')
df_ssim_loss_t00 = df_losses_t00[['global_batch_num', 'loss_pixel_t0.0']].set_index('global_batch_num')

################################
# t = 1.0
################################

ssim_report_file_t10 = os.path.join(log_dir_t10, 'MRI_GAN', 'ssim_report_file.pkl')
losses_file_t10 = os.path.join(log_dir_t10, 'MRI_GAN', 'losses.pkl')
metadata_file_t10 = os.path.join(log_dir_t10, 'MRI_GAN', 'mri_metadata.pkl')
losses_t10 = pickle.load(open(losses_file_t10, "rb"))
ssim_report_t10 = pickle.load(open(ssim_report_file_t10, "rb"))
metadata_t10 = pickle.load(open(metadata_file_t10, "rb"))
if metadata_t10['model_params']['tau'] == 1.0:
    print('metadata_t10 is valid')
else:
    raise Exception('Bad tau')

df_ssim_t10 = pd.DataFrame(data=ssim_report_t10,
                           columns=['epoch', 'local_batch_num', 'global_batch_num', 'mean_ssim_t1.0'])
df_ssim_t10 = df_ssim_t10[['global_batch_num', 'mean_ssim_t1.0']].set_index('global_batch_num')

df_losses_t10 = pd.DataFrame(data=losses_t10, columns=['epoch', 'local_batch_num', 'global_batch_num',
                                                       'loss_G_t1.0', 'loss_GAN_t1.0', 'loss_pixel_t1.0',
                                                       'loss_ssim_t1.0',
                                                       'loss_D_t1.0', 'loss_real_t1.0', 'loss_fake_t1.0'])
df_losses_t10 = df_losses_t10.drop(labels=['epoch', 'local_batch_num'], axis=1)  # .set_index('global_batch_num')
df_G_only_t10 = df_losses_t10[['global_batch_num', 'loss_G_t1.0']].set_index('global_batch_num')
df_D_only_t10 = df_losses_t10[['global_batch_num', 'loss_D_t1.0']].set_index('global_batch_num')
df_pixel_loss_t10 = df_losses_t10[['global_batch_num', 'loss_pixel_t1.0']].set_index('global_batch_num')
df_ssim_loss_t10 = df_losses_t10[['global_batch_num', 'loss_pixel_t1.0']].set_index('global_batch_num')

################################
# t = 0.3
################################
ssim_report_file_t03 = os.path.join(log_dir_t03, 'MRI_GAN', 'ssim_report_file.pkl')
losses_file_t03 = os.path.join(log_dir_t03, 'MRI_GAN', 'losses.pkl')
metadata_file_t03 = os.path.join(log_dir_t03, 'MRI_GAN', 'mri_metadata.pkl')
losses_t03 = pickle.load(open(losses_file_t03, "rb"))
ssim_report_t03 = pickle.load(open(ssim_report_file_t03, "rb"))
metadata_t03 = pickle.load(open(metadata_file_t03, "rb"))
if metadata_t03['model_params']['tau'] == 0.3:
    print('metadata_t03 is valid')
else:
    raise Exception('Bad tau')

df_ssim_t03 = pd.DataFrame(data=ssim_report_t03,
                           columns=['epoch', 'local_batch_num', 'global_batch_num', 'mean_ssim_t0.3'])
df_ssim_t03 = df_ssim_t03[['global_batch_num', 'mean_ssim_t0.3']].set_index('global_batch_num')

df_losses_t03 = pd.DataFrame(data=losses_t03, columns=['epoch', 'local_batch_num', 'global_batch_num',
                                                       'loss_G_t0.3', 'loss_GAN_t0.3', 'loss_pixel_t0.3',
                                                       'loss_ssim_t0.3',
                                                       'loss_D_t0.3', 'loss_real_t0.3', 'loss_fake_t0.3'])
df_losses_t03 = df_losses_t03.drop(labels=['epoch', 'local_batch_num'], axis=1)  # .set_index('global_batch_num')
df_G_only_t03 = df_losses_t03[['global_batch_num', 'loss_G_t0.3']].set_index('global_batch_num')
df_D_only_t03 = df_losses_t03[['global_batch_num', 'loss_D_t0.3']].set_index('global_batch_num')
df_pixel_loss_t03 = df_losses_t03[['global_batch_num', 'loss_pixel_t0.3']].set_index('global_batch_num')
df_ssim_loss_t03 = df_losses_t03[['global_batch_num', 'loss_pixel_t0.3']].set_index('global_batch_num')

################################
# t = 0.5
################################
ssim_report_file_t05 = os.path.join(log_dir_t05, 'MRI_GAN', 'ssim_report_file.pkl')
losses_file_t05 = os.path.join(log_dir_t05, 'MRI_GAN', 'losses.pkl')
metadata_file_t05 = os.path.join(log_dir_t05, 'MRI_GAN', 'mri_metadata.pkl')
losses_t05 = pickle.load(open(losses_file_t05, "rb"))
ssim_report_t05 = pickle.load(open(ssim_report_file_t05, "rb"))
metadata_t05 = pickle.load(open(metadata_file_t05, "rb"))
if metadata_t05['model_params']['tau'] == 0.5:
    print('metadata_t05 is valid')
else:
    raise Exception('Bad tau')

df_ssim_t05 = pd.DataFrame(data=ssim_report_t05,
                           columns=['epoch', 'local_batch_num', 'global_batch_num', 'mean_ssim_t0.5'])
df_ssim_t05 = df_ssim_t05[['global_batch_num', 'mean_ssim_t0.5']].set_index('global_batch_num')

df_losses_t05 = pd.DataFrame(data=losses_t05, columns=['epoch', 'local_batch_num', 'global_batch_num',
                                                       'loss_G_t0.5', 'loss_GAN_t0.5', 'loss_pixel_t0.5',
                                                       'loss_ssim_t0.5',
                                                       'loss_D_t0.5', 'loss_real_t0.5', 'loss_fake_t0.5'])
df_losses_t05 = df_losses_t05.drop(labels=['epoch', 'local_batch_num'], axis=1)
df_G_only_t05 = df_losses_t05[['global_batch_num', 'loss_G_t0.5']].set_index('global_batch_num')
df_D_only_t05 = df_losses_t05[['global_batch_num', 'loss_D_t0.5']].set_index('global_batch_num')
df_pixel_loss_t05 = df_losses_t05[['global_batch_num', 'loss_pixel_t0.5']].set_index('global_batch_num')
df_ssim_loss_t05 = df_losses_t05[['global_batch_num', 'loss_pixel_t0.5']].set_index('global_batch_num')

################################
# t = 0.7
################################

ssim_report_file_t07 = os.path.join(log_dir_t07, 'MRI_GAN', 'ssim_report_file.pkl')
losses_file_t07 = os.path.join(log_dir_t07, 'MRI_GAN', 'losses.pkl')
metadata_file_t07 = os.path.join(log_dir_t07, 'MRI_GAN', 'mri_metadata.pkl')
losses_t07 = pickle.load(open(losses_file_t07, "rb"))
ssim_report_t07 = pickle.load(open(ssim_report_file_t07, "rb"))
metadata_t07 = pickle.load(open(metadata_file_t07, "rb"))
if metadata_t07['model_params']['tau'] == 0.7:
    print('metadata_t07 is valid')
else:
    #raise Exception('Bad tau')
    pass

df_ssim_t07 = pd.DataFrame(data=ssim_report_t07,
                           columns=['epoch', 'local_batch_num', 'global_batch_num', 'mean_ssim_t0.7'])
df_ssim_t07 = df_ssim_t07[['global_batch_num', 'mean_ssim_t0.7']].set_index('global_batch_num')

df_losses_t07 = pd.DataFrame(data=losses_t07, columns=['epoch', 'local_batch_num', 'global_batch_num',
                                                       'loss_G_t0.7', 'loss_GAN_t0.7', 'loss_pixel_t0.7',
                                                       'loss_ssim_t0.7',
                                                       'loss_D_t0.7', 'loss_real_t0.7', 'loss_fake_t0.7'])
df_losses_t07 = df_losses_t07.drop(labels=['epoch', 'local_batch_num'], axis=1)  # .set_index('global_batch_num')
df_G_only_t07 = df_losses_t07[['global_batch_num', 'loss_G_t0.7']].set_index('global_batch_num')
df_D_only_t07 = df_losses_t07[['global_batch_num', 'loss_D_t0.7']].set_index('global_batch_num')
df_pixel_loss_t07 = df_losses_t07[['global_batch_num', 'loss_pixel_t0.7']].set_index('global_batch_num')
df_ssim_loss_t07 = df_losses_t07[['global_batch_num', 'loss_pixel_t0.7']].set_index('global_batch_num')

"""
################################
# common vars
################################
colors = ['tab:blue', 'tab:brown', 'tab:olive']
greek_letterz = [chr(code) for code in range(945, 970)]
s3 = greek_letterz[19] + '=0.3'
s5 = greek_letterz[19] + '=0.5'
s7 = greek_letterz[19] + '=0.7'
df_coloums_base = [s3, s5, s7]

################################
# plot SSIM score
################################

ax = plt.gca()
df_ssim_merge = [df_ssim_t03, df_ssim_t05, df_ssim_t07]
df_ssim = pd.concat(df_ssim_merge, join='outer', axis=1)  # .fillna(0)

df_ssim.columns = [s + ' actual' for s in df_coloums_base]
ax_ssim = df_ssim.plot(figsize=(16, 8), fontsize=tick_fontsize,
                       grid=True, alpha=0.3,
                       ax=ax, color=colors)  # title='SSIM score vs Global batches', )

ts_factor = 0.99
df_ssim_t03_smooth = df_ssim_t03.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_t05_smooth = df_ssim_t05.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_t07_smooth = df_ssim_t07.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_merge_smooth = [df_ssim_t03_smooth, df_ssim_t05_smooth, df_ssim_t07_smooth]
df_ssim_smooth = pd.concat(df_ssim_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_ssim_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_ssim_smooth = df_ssim_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                     grid=True, alpha=0.9, ax=ax,
                                     color=colors, linewidth=2)  # title='SSIM score vs Global batches')

ax_ssim_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_ssim_smooth.set_ylabel("Mean SSIM Score", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_ssim_smooth.get_xticks() / 1000]
ax_ssim_smooth.set_xticklabels(xlabels)
plt.legend(prop={"size": legend_fontsize})
fig = ax_ssim_smooth.get_figure()
fig.savefig('ssim_score_plot.png')
plt.clf()

################################
# plot Loss G
################################

ax = plt.gca()
df_loss_G_only_merge = [df_G_only_t03.reset_index(drop=True),
                        df_G_only_t05.reset_index(drop=True),
                        df_G_only_t07.reset_index(drop=True)]
df_loss_G_only = pd.concat(df_loss_G_only_merge, join='outer', axis=1)  # .fillna(0)

df_loss_G_only.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_G_only = df_loss_G_only.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                     grid=True, alpha=0.3,
                                     ax=ax, color=colors)  # title='loss_G_only score vs Global batches', )

ts_factor = 0.99
df_loss_G_only_t03_smooth = df_G_only_t03.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_t05_smooth = df_G_only_t05.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_t07_smooth = df_G_only_t07.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_merge_smooth = [df_loss_G_only_t03_smooth.reset_index(drop=True),
                               df_loss_G_only_t05_smooth.reset_index(drop=True),
                               df_loss_G_only_t07_smooth.reset_index(drop=True)]
df_loss_G_only_smooth = pd.concat(df_loss_G_only_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_loss_G_only_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_loss_G_only_smooth = df_loss_G_only_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                                   grid=True, alpha=0.9, ax=ax,
                                                   color=colors,
                                                   linewidth=2)  # title='loss_G_only score vs Global batches')

ax_loss_G_only_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_loss_G_only_smooth.set_ylabel("Total Generator Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_loss_G_only_smooth.get_xticks() / 1000]
ax_loss_G_only_smooth.set_xticklabels(xlabels)
plt.legend(prop={"size": legend_fontsize})
fig = ax_loss_G_only_smooth.get_figure()
fig.savefig('loss_G_plot.png')
plt.clf()

################################
# plot Loss D
################################

ax = plt.gca()
df_loss_D_only_merge = [df_D_only_t03.reset_index(drop=True),
                        df_D_only_t05.reset_index(drop=True),
                        df_D_only_t07.reset_index(drop=True)]
df_loss_D_only = pd.concat(df_loss_D_only_merge, join='outer', axis=1)  # .fillna(0)

df_loss_D_only.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_D_only = df_loss_D_only.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                     grid=True, alpha=0.3,
                                     ax=ax, color=colors)
ax_loss_D_only.set_ylim(0, 0.50)
ts_factor = 0.99
df_loss_D_only_t03_smooth = df_D_only_t03.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_t05_smooth = df_D_only_t05.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_t07_smooth = df_D_only_t07.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_merge_smooth = [df_loss_D_only_t03_smooth.reset_index(drop=True),
                               df_loss_D_only_t05_smooth.reset_index(drop=True),
                               df_loss_D_only_t07_smooth.reset_index(drop=True)]
df_loss_D_only_smooth = pd.concat(df_loss_D_only_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_loss_D_only_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_loss_D_only_smooth = df_loss_D_only_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                                   grid=True, alpha=0.9, ax=ax,
                                                   color=colors,
                                                   linewidth=2)

ax_loss_D_only_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_loss_D_only_smooth.set_ylabel("Total Discriminator Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_loss_D_only_smooth.get_xticks() / 1000]
ax_loss_D_only_smooth.set_xticklabels(xlabels)
ax_loss_D_only_smooth.set_ylim(0, 0.50)
plt.legend(prop={"size": legend_fontsize})
fig = ax_loss_D_only_smooth.get_figure()
fig.savefig('loss_D_plot.png')
plt.clf()

################################
# plot Loss G_ssim
################################

ax = plt.gca()
df_ssim_loss_merge = [df_ssim_loss_t03.reset_index(drop=True),
                      df_ssim_loss_t05.reset_index(drop=True),
                      df_ssim_loss_t07.reset_index(drop=True)]
df_loss_ssim = pd.concat(df_ssim_loss_merge, join='outer', axis=1)  # .fillna(0)

df_loss_ssim.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_ssim = df_loss_ssim.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                 grid=True, alpha=0.3,
                                 ax=ax, color=colors)
ax_loss_ssim.set_ylim(0, 0.50)
ts_factor = 0.99
df_ssim_loss_t03_smooth = df_ssim_loss_t03.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_t05_smooth = df_ssim_loss_t05.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_t07_smooth = df_ssim_loss_t07.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_merge_smooth = [df_ssim_loss_t03_smooth.reset_index(drop=True),
                             df_ssim_loss_t05_smooth.reset_index(drop=True),
                             df_ssim_loss_t07_smooth.reset_index(drop=True)]
df_ssim_loss_smooth = pd.concat(df_ssim_loss_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_ssim_loss_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_ssim_loss_smooth = df_ssim_loss_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                               grid=True, alpha=0.9, ax=ax,
                                               color=colors,
                                               linewidth=2)

ax_ssim_loss_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_ssim_loss_smooth.set_ylabel("SSIM Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_ssim_loss_smooth.get_xticks() / 1000]
ax_ssim_loss_smooth.set_xticklabels(xlabels)
ax_ssim_loss_smooth.set_ylim(0, 0.50)
plt.legend(prop={"size": legend_fontsize})
fig = ax_ssim_loss_smooth.get_figure()
fig.savefig('ssim_loss_plot.png')
plt.clf()

################################
# plot Loss G_pixel
################################
ax = plt.gca()
df_pixel_loss_merge = [df_pixel_loss_t03.reset_index(drop=True),
                       df_pixel_loss_t05.reset_index(drop=True),
                       df_pixel_loss_t07.reset_index(drop=True)]
df_loss_ssim = pd.concat(df_pixel_loss_merge, join='outer', axis=1)  # .fillna(0)

df_loss_ssim.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_ssim = df_loss_ssim.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                 grid=True, alpha=0.3,
                                 ax=ax, color=colors)
ax_loss_ssim.set_ylim(0, 0.50)
ts_factor = 0.99
df_pixel_loss_t03_smooth = df_pixel_loss_t03.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_t05_smooth = df_pixel_loss_t05.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_t07_smooth = df_pixel_loss_t07.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_merge_smooth = [df_pixel_loss_t03_smooth.reset_index(drop=True),
                              df_pixel_loss_t05_smooth.reset_index(drop=True),
                              df_pixel_loss_t07_smooth.reset_index(drop=True)]
df_pixel_loss_smooth = pd.concat(df_pixel_loss_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_pixel_loss_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_pixel_loss_smooth = df_pixel_loss_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                                 grid=True, alpha=0.9, ax=ax,
                                                 color=colors,
                                                 linewidth=2)

ax_pixel_loss_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_pixel_loss_smooth.set_ylabel(r"L$_2$ Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_pixel_loss_smooth.get_xticks() / 1000]
ax_pixel_loss_smooth.set_xticklabels(xlabels)
ax_pixel_loss_smooth.set_ylim(0, 0.50)
plt.legend(prop={"size": legend_fontsize})
fig = ax_pixel_loss_smooth.get_figure()
fig.savefig('L2_loss_plot.png')
plt.clf()

"""

################################
# common vars
################################
colors = ['tab:purple', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:olive']
greek_letterz = [chr(code) for code in range(945, 970)]
s0 = greek_letterz[19] + '=0.0'
s1 = greek_letterz[19] + '=1.0'
s3 = greek_letterz[19] + '=0.3'
s5 = greek_letterz[19] + '=0.5'
s7 = greek_letterz[19] + '=0.7'
df_coloums_base = [s0, s1, s3, s5, s7]

################################
# plot SSIM score
################################

ax = plt.gca()
df_ssim_merge = [df_ssim_t00, df_ssim_t10, df_ssim_t03, df_ssim_t05, df_ssim_t07]
df_ssim = pd.concat(df_ssim_merge, join='outer', axis=1)  # .fillna(0)

df_ssim.columns = [s + ' actual' for s in df_coloums_base]
ax_ssim = df_ssim.plot(figsize=(16, 8), fontsize=tick_fontsize,
                       grid=True, alpha=0.3,
                       ax=ax, color=colors)  # title='SSIM score vs Global batches', )

ts_factor = 0.99
df_ssim_t00_smooth = df_ssim_t00.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_t10_smooth = df_ssim_t10.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_t03_smooth = df_ssim_t03.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_t05_smooth = df_ssim_t05.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_t07_smooth = df_ssim_t07.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_merge_smooth = [df_ssim_t00_smooth, df_ssim_t10_smooth, df_ssim_t03_smooth, df_ssim_t05_smooth,
                        df_ssim_t07_smooth]
df_ssim_smooth = pd.concat(df_ssim_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_ssim_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_ssim_smooth = df_ssim_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                     grid=True, alpha=0.9, ax=ax,
                                     color=colors, linewidth=2)  # title='SSIM score vs Global batches')

ax_ssim_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_ssim_smooth.set_ylabel("Mean SSIM Score", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_ssim_smooth.get_xticks() / 1000]
ax_ssim_smooth.set_xticklabels(xlabels)
plt.legend(prop={"size": legend_fontsize})
fig = ax_ssim_smooth.get_figure()
fig.savefig('ssim_score_plot.png')
plt.clf()

################################
# plot Loss G
################################

ax = plt.gca()
df_loss_G_only_merge = [df_G_only_t00.reset_index(drop=True),
                        df_G_only_t10.reset_index(drop=True),
                        df_G_only_t03.reset_index(drop=True),
                        df_G_only_t05.reset_index(drop=True),
                        df_G_only_t07.reset_index(drop=True)]
df_loss_G_only = pd.concat(df_loss_G_only_merge, join='outer', axis=1)  # .fillna(0)

df_loss_G_only.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_G_only = df_loss_G_only.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                     grid=True, alpha=0.3,
                                     ax=ax, color=colors)  # title='loss_G_only score vs Global batches', )

ts_factor = 0.99
df_loss_G_only_t00_smooth = df_G_only_t00.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_t10_smooth = df_G_only_t10.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_t03_smooth = df_G_only_t03.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_t05_smooth = df_G_only_t05.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_t07_smooth = df_G_only_t07.ewm(alpha=(1 - ts_factor)).mean()
df_loss_G_only_merge_smooth = [df_loss_G_only_t00_smooth.reset_index(drop=True),
                               df_loss_G_only_t10_smooth.reset_index(drop=True),
                               df_loss_G_only_t03_smooth.reset_index(drop=True),
                               df_loss_G_only_t05_smooth.reset_index(drop=True),
                               df_loss_G_only_t07_smooth.reset_index(drop=True)]
df_loss_G_only_smooth = pd.concat(df_loss_G_only_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_loss_G_only_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_loss_G_only_smooth = df_loss_G_only_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                                   grid=True, alpha=0.9, ax=ax,
                                                   color=colors,
                                                   linewidth=2)  # title='loss_G_only score vs Global batches')

ax_loss_G_only_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_loss_G_only_smooth.set_ylabel("Total Generator Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_loss_G_only_smooth.get_xticks() / 1000]
ax_loss_G_only_smooth.set_xticklabels(xlabels)
plt.legend(prop={"size": legend_fontsize})
fig = ax_loss_G_only_smooth.get_figure()
fig.savefig('loss_G_plot.png')
plt.clf()

################################
# plot Loss D
################################

ax = plt.gca()
df_loss_D_only_merge = [df_D_only_t00.reset_index(drop=True),
                        df_D_only_t10.reset_index(drop=True),
                        df_D_only_t03.reset_index(drop=True),
                        df_D_only_t05.reset_index(drop=True),
                        df_D_only_t07.reset_index(drop=True)]
df_loss_D_only = pd.concat(df_loss_D_only_merge, join='outer', axis=1)  # .fillna(0)

df_loss_D_only.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_D_only = df_loss_D_only.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                     grid=True, alpha=0.3,
                                     ax=ax, color=colors)
ax_loss_D_only.set_ylim(0, 0.50)
ts_factor = 0.99
df_loss_D_only_t00_smooth = df_D_only_t00.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_t10_smooth = df_D_only_t10.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_t03_smooth = df_D_only_t03.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_t05_smooth = df_D_only_t05.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_t07_smooth = df_D_only_t07.ewm(alpha=(1 - ts_factor)).mean()
df_loss_D_only_merge_smooth = [df_loss_D_only_t00_smooth.reset_index(drop=True),
                               df_loss_D_only_t10_smooth.reset_index(drop=True),
                               df_loss_D_only_t03_smooth.reset_index(drop=True),
                               df_loss_D_only_t05_smooth.reset_index(drop=True),
                               df_loss_D_only_t07_smooth.reset_index(drop=True)]
df_loss_D_only_smooth = pd.concat(df_loss_D_only_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_loss_D_only_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_loss_D_only_smooth = df_loss_D_only_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                                   grid=True, alpha=0.9, ax=ax,
                                                   color=colors,
                                                   linewidth=2)

ax_loss_D_only_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_loss_D_only_smooth.set_ylabel("Total Discriminator Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_loss_D_only_smooth.get_xticks() / 1000]
ax_loss_D_only_smooth.set_xticklabels(xlabels)
ax_loss_D_only_smooth.set_ylim(0, 0.50)
plt.legend(prop={"size": legend_fontsize})
fig = ax_loss_D_only_smooth.get_figure()
fig.savefig('loss_D_plot.png')
plt.clf()

################################
# plot Loss G_ssim
################################

ax = plt.gca()
df_ssim_loss_merge = [df_ssim_loss_t00.reset_index(drop=True),
                      df_ssim_loss_t10.reset_index(drop=True),
                      df_ssim_loss_t03.reset_index(drop=True),
                      df_ssim_loss_t05.reset_index(drop=True),
                      df_ssim_loss_t07.reset_index(drop=True)]
df_loss_ssim = pd.concat(df_ssim_loss_merge, join='outer', axis=1)  # .fillna(0)

df_loss_ssim.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_ssim = df_loss_ssim.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                 grid=True, alpha=0.3,
                                 ax=ax, color=colors)
ax_loss_ssim.set_ylim(0, 0.50)
ts_factor = 0.99
df_ssim_loss_t10_smooth = df_ssim_loss_t10.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_t00_smooth = df_ssim_loss_t00.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_t03_smooth = df_ssim_loss_t03.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_t05_smooth = df_ssim_loss_t05.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_t07_smooth = df_ssim_loss_t07.ewm(alpha=(1 - ts_factor)).mean()
df_ssim_loss_merge_smooth = [df_ssim_loss_t00_smooth.reset_index(drop=True),
                             df_ssim_loss_t10_smooth.reset_index(drop=True),
                             df_ssim_loss_t03_smooth.reset_index(drop=True),
                             df_ssim_loss_t05_smooth.reset_index(drop=True),
                             df_ssim_loss_t07_smooth.reset_index(drop=True)]
df_ssim_loss_smooth = pd.concat(df_ssim_loss_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_ssim_loss_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_ssim_loss_smooth = df_ssim_loss_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                               grid=True, alpha=0.9, ax=ax,
                                               color=colors,
                                               linewidth=2)

ax_ssim_loss_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_ssim_loss_smooth.set_ylabel("SSIM Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_ssim_loss_smooth.get_xticks() / 1000]
ax_ssim_loss_smooth.set_xticklabels(xlabels)
ax_ssim_loss_smooth.set_ylim(0, 0.50)
plt.legend(prop={"size": legend_fontsize})
fig = ax_ssim_loss_smooth.get_figure()
fig.savefig('ssim_loss_plot.png')
plt.clf()

################################
# plot Loss G_pixel
################################
ax = plt.gca()
df_pixel_loss_merge = [df_pixel_loss_t00.reset_index(drop=True),
                       df_pixel_loss_t10.reset_index(drop=True),
                       df_pixel_loss_t03.reset_index(drop=True),
                       df_pixel_loss_t05.reset_index(drop=True),
                       df_pixel_loss_t07.reset_index(drop=True)]
df_loss_ssim = pd.concat(df_pixel_loss_merge, join='outer', axis=1)  # .fillna(0)

df_loss_ssim.columns = [s + ' actual' for s in df_coloums_base]
ax_loss_ssim = df_loss_ssim.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                 grid=True, alpha=0.3,
                                 ax=ax, color=colors)
ax_loss_ssim.set_ylim(0, 0.50)
ts_factor = 0.99
df_pixel_loss_t00_smooth = df_pixel_loss_t00.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_t10_smooth = df_pixel_loss_t10.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_t03_smooth = df_pixel_loss_t03.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_t05_smooth = df_pixel_loss_t05.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_t07_smooth = df_pixel_loss_t07.ewm(alpha=(1 - ts_factor)).mean()
df_pixel_loss_merge_smooth = [df_pixel_loss_t00_smooth.reset_index(drop=True),
                              df_pixel_loss_t10_smooth.reset_index(drop=True),
                              df_pixel_loss_t03_smooth.reset_index(drop=True),
                              df_pixel_loss_t05_smooth.reset_index(drop=True),
                              df_pixel_loss_t07_smooth.reset_index(drop=True)]
df_pixel_loss_smooth = pd.concat(df_pixel_loss_merge_smooth, join='outer', axis=1)  # .fillna(0)
df_pixel_loss_smooth.columns = [s + ' smoothen' for s in df_coloums_base]
ax_pixel_loss_smooth = df_pixel_loss_smooth.plot(figsize=(16, 8), fontsize=tick_fontsize,
                                                 grid=True, alpha=0.9, ax=ax,
                                                 color=colors,
                                                 linewidth=2)

ax_pixel_loss_smooth.set_xlabel("Global batch Number", fontsize=xylabel_fontsize)
ax_pixel_loss_smooth.set_ylabel(r"L$_2$ Loss", fontsize=xylabel_fontsize)
xlabels = ['{}'.format(x) + 'k' for x in ax_pixel_loss_smooth.get_xticks() / 1000]
ax_pixel_loss_smooth.set_xticklabels(xlabels)
ax_pixel_loss_smooth.set_ylim(0, 0.50)
plt.legend(prop={"size": legend_fontsize})
fig = ax_pixel_loss_smooth.get_figure()
fig.savefig('L2_loss_plot.png')
plt.clf()
