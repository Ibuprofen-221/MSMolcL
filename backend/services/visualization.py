import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import List, Dict, Any, Optional
import os

# RDKit 用于 2D 分子结构
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class SpectrumVisualizer:
    """谱图可视化器 (本地绘图版)"""

    @staticmethod
    def plot_spectrum_to_file(spectrum_data: Dict[str, Any], 
                             output_path: str = "spectrum.png",
                             title: str = "Mass Spectrum"):
        """
        绘制质谱图并保存为图片
        """
        peaks = spectrum_data.get("peaks")
        if peaks is None or len(peaks) == 0:
            print(f"Warning: {title} has no peak data.")
            return

        # 数据提取逻辑
        if hasattr(peaks, 'mz') and hasattr(peaks, 'intensities'):
            mz = peaks.mz
            intensity = peaks.intensities
        else:
            mz = peaks[:, 0]
            intensity = peaks[:, 1]

        # 使用 Matplotlib 进行静态绘图（更适合后台脚本生成图片）
        plt.figure(figsize=(10, 6))
        max_intensity = intensity.max()
        
        # 绘制针状图 (Stem plot)
        markerline, stemlines, baseline = plt.stem(mz, intensity, basefmt=" ", markerfmt=" ")
        plt.setp(stemlines, 'color', '#4F46E5', 'linewidth', 1.5)
        
        # 标注主要峰的 m/z 值
        for i, (m, inv) in enumerate(zip(mz, intensity)):
            if inv > max_intensity * 0.2:  # 只标注强度大于20%的峰
                plt.text(m, inv + (max_intensity * 0.02), f'{m:.2f}', 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.title(title, fontsize=14, color='#1E40AF', fontweight='bold')
        plt.xlabel('m/z', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Spectrum saved to: {output_path}")


class MoleculeVisualizer:
    """分子结构可视化器 (本地绘图版)"""

    @staticmethod
    def save_molecule_2d(smiles: str, 
                         output_path: str = "molecule.png", 
                         size: tuple = (600, 600)):
        """
        将 SMILES 转换为图片文件
        """
        if not RDKIT_AVAILABLE:
            print("Error: RDKit is not installed.")
            return

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Error: Invalid SMILES: {smiles}")
                return

            # 计算基础元数据
            mw = Descriptors.ExactMolWt(mol)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

            # 绘图
            img = Draw.MolToImage(mol, size=size, kekulize=True, wedgeBonds=True)
            
            # 在图片下方添加标注（可选）
            img.save(output_path)
            print(f"Molecule image saved to: {output_path} (MW: {mw:.4f}, Formula: {formula})")

        except Exception as e:
            print(f"Failed to generate molecule image: {e}")


def main(
    smiles_input: str, 
    spectrum_peaks: np.ndarray, 
    output_dir: str = "output_results"
):
    """
    显式输入参数的执行函数
    
    Args:
        smiles_input: 分子的 SMILES 字符串
        spectrum_peaks: 谱图峰数据, numpy 数组格式 [[mz, intensity], ...]
        output_dir: 图片保存目录
    """
    # 1. 准备环境
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 处理谱图可视化
    spectrum_data = {"peaks": spectrum_peaks}
    spec_path = os.path.join(output_dir, "spectrum_plot.png")
    SpectrumVisualizer.plot_spectrum_to_file(
        spectrum_data, 
        output_path=spec_path, 
        title="Analysis Result Spectrum"
    )

    # 3. 处理分子结构可视化
    mol_path = os.path.join(output_dir, "molecule_structure.png")
    MoleculeVisualizer.save_molecule_2d(
        smiles=smiles_input, 
        output_path=mol_path
    )

    print("\nProcessing complete.")


if __name__ == "__main__":
    # --- 显式输入参数示例 ---
    
    # 1. 模拟输入的 SMILES (例如：阿司匹林)
    example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # 2. 模拟输入的谱图数据 (m/z, intensity)
    example_peaks = np.array([
        [120.04, 100.0],
        [138.05, 45.2],
        [163.04, 12.5],
        [180.06, 88.3],
        [92.03, 30.1]
    ])

    # 3. 调用主函数
    main(
        smiles_input=example_smiles, 
        spectrum_peaks=example_peaks, 
        output_dir="my_analysis_report"
    )