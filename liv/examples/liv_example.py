import clip 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch 
import torchvision.transforms as T
from PIL import Image 
from liv import load_liv

model = load_liv()
model.eval()
transform = T.Compose([T.ToTensor()])

task = "open microwave"

imgs = []
imgs_tensor = []
start_frame = 33545
end_frame = 33601
for index in range(start_frame, end_frame):
    img = Image.open(f"sample_video/frame_0000{index+1:06}.jpg")
    imgs.append(img)
    imgs_tensor.append(transform(img))
imgs_tensor = torch.stack(imgs_tensor)
with torch.no_grad():
    embeddings = model(input=imgs_tensor.cuda(), modality="vision")
    goal_embedding_img = embeddings[-1]
    token = clip.tokenize([task])
    goal_embedding_text = model(input=token, modality="text")
    goal_embedding_text = goal_embedding_text[0] 

fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(24,6))
distances_cur_img = []
distances_cur_text = [] 
for t in range(embeddings.shape[0]):
    cur_embedding = embeddings[t]
    cur_distance_img = - model.module.sim(goal_embedding_img, cur_embedding).detach().cpu().numpy()
    cur_distance_text = - model.module.sim(goal_embedding_text, cur_embedding).detach().cpu().numpy()

    distances_cur_img.append(cur_distance_img)
    distances_cur_text.append(cur_distance_text)

distances_cur_img = np.array(distances_cur_img)
distances_cur_text = np.array(distances_cur_text)

ax[0].plot(np.arange(len(distances_cur_img)), distances_cur_img, color="tab:blue", label="image", linewidth=3)
ax[0].plot(np.arange(len(distances_cur_text)), distances_cur_text, color="tab:red", label="text", linewidth=3)
ax[1].plot(np.arange(len(distances_cur_img)), distances_cur_img, color="tab:blue", label="image", linewidth=3)
ax[2].plot(np.arange(len(distances_cur_text)), distances_cur_text, color="tab:red", label="text", linewidth=3)

ax[0].legend(loc="upper right")
ax[0].set_xlabel("Frame", fontsize=15)
ax[1].set_xlabel("Frame", fontsize=15)
ax[2].set_xlabel("Frame", fontsize=15)
ax[0].set_ylabel("Embedding Distance", fontsize=15)
ax[0].set_title(f"Language Goal: {task}", fontsize=15)
ax[1].set_title("Image Goal", fontsize=15)
ax[2].set_title(f"Language Goal: {task}", fontsize=15)
ax[3].imshow(imgs_tensor[-1].permute(1,2,0))
asp = 1
ax[0].set_aspect(asp * np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
ax[1].set_aspect(asp * np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
ax[2].set_aspect(asp * np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0])
fig.savefig(f"liv_open_microwave.png", bbox_inches='tight')
plt.close()

ax0_xlim = ax[0].get_xlim()
ax0_ylim = ax[0].get_ylim()
ax1_xlim = ax[1].get_xlim()
ax1_ylim = ax[1].get_ylim()
ax2_xlim = ax[2].get_xlim()
ax2_ylim = ax[2].get_ylim()

def animate(i):
    for ax_subplot in ax:
        ax_subplot.clear()
    ranges = np.arange(len(distances_cur_img))
    if i >= len(distances_cur_img):
        i = len(distances_cur_img) - 1
    line1 = ax[0].plot(ranges[0:i], distances_cur_img[0:i], color="tab:blue", label="image", linewidth=3)
    line2 = ax[0].plot(ranges[0:i], distances_cur_text[0:i], color="tab:red", label="text", linewidth=3)
    line3 = ax[1].plot(ranges[0:i], distances_cur_img[0:i], color="tab:blue", label="image", linewidth=3)
    line4 = ax[2].plot(ranges[0:i], distances_cur_text[0:i], color="tab:red", label="text", linewidth=3)
    line5 = ax[3].imshow(imgs_tensor[i].permute(1,2,0))
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[1].set_xlabel("Frame", fontsize=15)
    ax[2].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Embedding Distance", fontsize=15)
    ax[0].set_title(f"Language Goal: {task}", fontsize=15)
    ax[1].set_title("Image Goal", fontsize=15)
    ax[2].set_title(f"Language Goal: {task}", fontsize=15)

    ax[0].set_xlim(ax0_xlim)
    ax[0].set_ylim(ax0_ylim)
    ax[1].set_xlim(ax1_xlim)
    ax[1].set_ylim(ax1_ylim)
    ax[2].set_xlim(ax2_xlim)
    ax[2].set_ylim(ax2_ylim)

    return line1, line2, line3, line4, line5

# Generate animated reward curve
ani = FuncAnimation(fig, animate, interval=20, repeat=False, frames=len(distances_cur_img)+30)    
ani.save(f"liv_open_microwave.gif", dpi=100, writer=PillowWriter(fps=25))