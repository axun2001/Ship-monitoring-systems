from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

# matplotlib reference:
# http://pynote.hatenablog.com/entry/matplotlib-surface-plot
# https://qiita.com/kazetof/items/c0204f197d394458022a
plt.rcParams['font.family'] = 'Times New Roman'

def visualize():

    vmin = 0
    vmax = 100
    vlevel = 0.5
    result_file_path = "3d_surface_test_adv_file.h5"
    surf_name = "test_loss"
    colors = [(2 / 255, 48 / 255, 71 / 255), (14 / 255, 91 / 255, 118 / 255),
              (26 / 255, 134 / 255, 163 / 255), (70 / 255, 172 / 255, 202 / 255),
              (155 / 255, 207 / 255, 232 / 255), (255 / 255, 202 / 255, 95 / 255),
              (254 / 255, 168 / 255, 9 / 255), (251 / 255, 132 / 255, 2 / 255)]

    positions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]

    # 创建颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    with h5py.File(result_file_path,'r') as f:
        # 遍历文件中的数据集
        print("数据集:")
        for dataset_name in f:
            print(dataset_name)

        # 遍历文件中的组
        print("\n组:")

        def print_groups(name, obj):
            if isinstance(obj, h5py.Group):
                print(name)

        f.visititems(print_groups)

        Z_LIMIT = 100

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])

        X, Y = np.meshgrid(x, y)
        Z = np.array(f[surf_name][:])
        Z[Z > Z_LIMIT] = Z_LIMIT
        Z = np.log(Z)  # logscale

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1,1)
        #ax.set_zlim3d(-5,8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #ax.plot_wireframe(X, Y, Z)

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        # Save 2D contours image
        fig = plt.figure()
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        CS = plt.contour(X, Y, Z, cmap=cmap, levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        plt.show()
        fig.savefig(result_file_path + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')



        fig = plt.figure()
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        CS = plt.contourf(X, Y, Z, cmap=cmap, levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        plt.show()
        fig.savefig(result_file_path + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        # Save 2D heatmaps image
        plt.figure()
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        sns_plot = sns.heatmap(Z, cmap=cmap, cbar=True, vmin=vmin, vmax=vmax,
                               xticklabels=False, yticklabels=False)
        sns_plot.invert_yaxis()
        sns_plot.get_figure().savefig(result_file_path + '_' + surf_name + '_2dheat.pdf',
                                      dpi=300, bbox_inches='tight', format='pdf')
        plt.show()

        # Save 3D surface image
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True)
        fig.savefig(result_file_path + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')
        plt.show()

visualize()
