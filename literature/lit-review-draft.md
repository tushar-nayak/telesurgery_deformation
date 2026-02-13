# Literature Review Draft: Physics-Informed Neural Networks for Guidewire-Induced Vascular Deformation Estimation

**Author:** Tushar Nayak  
**Course:** Technical Writing for Biomedical Engineering  
**Date:** February 13, 2026

---

## Introduction: The Challenge of Intraoperative Vascular Deformation

Endovascular thrombectomy has revolutionized the treatment of acute ischemic stroke and other vascular pathologies, offering minimally invasive alternatives to open surgery with reduced patient recovery times and procedural risks. During these procedures, interventionalists navigate guidewires, catheters, and retrieval devices through complex cerebrovascular or coronary anatomy under real-time X-ray fluoroscopy guidance. However, the mechanical interaction between these devices and vessel walls induces significant anatomical deformation—vessels bend, stretch, and displace in response to device forces—rendering pre-operative computed tomography angiography (CTA) models inaccurate for real-time navigation[cite:8]. Surgeons must continuously reconstruct three-dimensional anatomy from limited two-dimensional fluoroscopic projections while simultaneously monitoring device position, assessing vessel morphology, and planning next maneuvers. This cognitive burden contributes to procedural complexity, prolonged fluoroscopy times, increased radiation exposure, and elevated risk of complications such as vessel perforation or dissection.

The core technical challenge lies in estimating **dynamic vessel deformation** from sparse, noisy, and projection-ambiguous intraoperative imaging. Unlike soft tissue organs (liver, brain) where deformation follows bulk displacement patterns, vascular structures exhibit high-curvature tubular geometry with radii ranging from 0.5–5 mm, making them sensitive to quantization errors and requiring specialized reconstruction approaches. Moreover, deformations must be estimated in real-time (<100 ms latency) to provide actionable surgical guidance, yet must respect biomechanical plausibility—vessels cannot kink beyond elastic limits, stretch infinitely, or exhibit discontinuous topology. Current image-guided surgery systems either rely on rigid registration of pre-operative models (failing under deformation) or employ purely data-driven methods that lack physical constraints and generalize poorly to unseen anatomies.

This literature review surveys the status quo in three intersecting domains: (1) **real-time intraoperative vessel tracking and registration**, (2) **physics-informed deformation modeling**, and (3) **sparse-view 3D vascular reconstruction**. By synthesizing advances and limitations across these fields, we identify a critical gap: the absence of computationally efficient, physics-constrained methods for estimating guidewire-induced vascular deformation from contrast-free fluoroscopy. Our proposed contribution—a sparse control point radial basis function (RBF) framework anchored in strain energy regularization—addresses this gap by enabling real-time, biomechanically plausible deformation estimation for intraoperative surgical assistance.

---

## The State of the Art: Advances and Limitations

### Real-Time Intraoperative Vessel Tracking Under Deformation

The most directly relevant prior work comes from interventional guidance systems that track instruments and compensate for anatomical motion. Yang et al.[cite:8] recently introduced a deformable instrument tip tracking (DITT) framework for transjugular intrahepatic portosystemic shunt (TIPS) procedures, achieving real-time 3D guidewire tip localization at 43 ms per frame with tracking errors of 1.59±0.57 mm in simulation and 1.67±0.54 mm in clinical datasets. Their system addresses three sources of anatomical change: (1) patient pose variation through adaptive 3D-2D centerline registration using hidden Markov models, (2) respiratory motion via learned liver displacement models from monoplane X-ray sequences, and (3) puncture-induced vessel deformation through non-rigid pointset registration with manifold regularization. Critically, Yang et al. track the *instrument tip position relative to a deforming vessel*, rather than estimating the *vessel wall deformation itself*. Their respiratory compensation module learns motion priors from liver mask sequences (extracted via CycleGAN domain adaptation and Dense-UNet segmentation), providing a template for modeling physiological motion. However, device-induced deformation—where the guidewire actively pushes, bends, and stretches the vessel—requires fundamentally different biomechanical modeling than passive respiratory displacement.

The distinction is crucial: respiratory motion exhibits periodic, low-frequency (<0.5 Hz) bulk translation of entire vascular trees, whereas guidewire interaction produces localized, high-curvature deformations concentrated at contact points. Yang et al.'s manifold regularization enforces geometric smoothness during non-rigid registration, but does not encode vascular tissue mechanics (e.g., strain energy, elastic limits). Their method also targets hepatic vasculature in TIPS procedures—where vessels are relatively large (3–10 mm diameter) and embedded in deformable liver parenchyma—differing from the slender, high-curvature cerebrovascular or coronary vessels encountered in thrombectomy. Nonetheless, their adaptive sampling strategy for 3D-2D feature correspondence and their demonstration of real-time performance on clinical data establish a critical benchmark for intraoperative deformation tracking.

Beyond real-time tracking, several works have addressed intraoperative 2D-3D registration for image-guided surgery, though typically without deformation modeling. Miao et al. pioneered CNN-based regression for real-time 2D-3D registration, while Schaffert et al. introduced attention mechanisms for robust point-to-plane correspondence in X-ray-to-CT alignment. However, these methods assume rigid transformations or rely on post-hoc non-rigid refinement steps that lack physical constraints. The absence of biomechanical regularization allows anatomically implausible deformations—vessels that fold impossibly, stretch beyond physiological limits, or exhibit discontinuous topology—undermining surgical decision-making.

### Physics-Informed Neural Networks for Deformable Registration

A parallel line of research has integrated continuum mechanics into deep learning frameworks, seeking to constrain learned deformations to physically plausible manifolds. Amiri-Hezaveh et al.[cite:5] proposed a predictor-corrector framework for deformable medical image registration that combines geometric alignment (predictor) with large-deformation elasticity equations (corrector). Their method employs neural ordinary differential equations (Neural ODEs) to guarantee diffeomorphic transformations—smooth, invertible mappings essential for tracking anatomical correspondence across time. The corrector step minimizes potential energy subject to equilibrium equations, optionally incorporating growth and remodeling phenomena through multiplicative decomposition of the deformation gradient (F = F_elastic · F_growth). Validated on zebrafish wound healing, brain atrophy, and fetal brain development, their framework demonstrates that physics-based regularization improves registration accuracy and biological interpretability.

However, Amiri-Hezaveh et al. address *image-to-image registration*—aligning two volumetric scans of the same anatomy at different time points—rather than estimating real-time deformation from 2D projections. Their training requires ground-truth 3D segmentations and volumetric masks, which are unavailable during live fluoroscopy. The corrector optimization, while ensuring energy minimization, incurs computational cost (100,000 training epochs per anatomy), limiting real-time applicability. Moreover, their focus on bulk tissue deformation (zebrafish tails, brain parenchyma) does not directly translate to slender vascular structures, where localized curvature and centerline topology dominate mechanical behavior.

Harper et al.[cite:6] addressed real-time requirements by embedding finite element method (FEM) elasticity constraints directly into neural network loss functions, terming their approach a physics-informed neural network (PINN) for augmented reality surgical tracking. Validated on liver and brain phantoms undergoing 20 mm induced deformations, their model achieved 1.1 mm mean registration error at 22 fps by penalizing predictions that violate biomechanical priors. Unlike Amiri-Hezaveh's iterative energy minimization, Harper's approach performs single-pass feed-forward inference, trading optimality for speed. However, their method relies on depth sensor input (structured-light scanners) rather than 2D X-ray projections, and targets surface-based registration of solid organs rather than centerline-based vascular structures. The strain energy formulations developed for bulk soft tissue (neo-Hookean, Mooney-Rivlin constitutive models) do not directly apply to thin-walled vessels, which require bending-dominated beam models.

The broader PINN literature, pioneered by Raissi et al., has successfully incorporated partial differential equations (PDEs) as soft constraints during training, enabling learning from sparse, noisy data while respecting conservation laws. Yet, medical applications of PINNs have predominantly focused on image reconstruction (CT, MRI) or fluid dynamics (blood flow simulation), with limited exploration of interventional guidance scenarios where *forward models are known* (projection geometry, tissue mechanics) but *boundary conditions are unknown* (device forces, contact locations).

### Sparse-View 3D Vascular Reconstruction

Reconstructing three-dimensional vascular anatomy from limited X-ray views has long been pursued for dose reduction and real-time guidance. Traditional approaches include iterative optimization (photometric consistency, epipolar constraints) and model-based methods (deformable templates, centerline fitting). Recently, deep learning has enabled learning-based reconstruction from as few as two X-ray projections. Zhu et al.[cite:9] introduced AutoCAR, a self-supervised sparse backwards projection (SBP) framework that reconstructs coronary arteries from biplane angiography without requiring 3D ground truth. Their method leverages domain adaptation (CTA-to-DRR-to-real-XA) to train neural networks that "unproject" 2D vessel masks into sparse 3D occupancy volumes, followed by graph optimization to enforce vascular topology and temporal coherence. Evaluated on 894 CTA volumes and 1,215 real angiography videos, AutoCAR achieved 92% coverage, 1.8 mm endpoint error, and 83% correspondence accuracy—matching experienced clinicians but at two orders of magnitude faster speed (7 seconds vs. several minutes).

AutoCAR's success demonstrates the feasibility of learning 3D-from-2D mappings for vascular structures using only synthetic training data, provided sufficient domain randomization and multi-view consistency. However, AutoCAR targets *static* vascular anatomy at a single cardiac phase, relying on contrast-enhanced angiography to visualize lumen geometry. Intraoperative fluoroscopy during guidewire manipulation is typically contrast-free (to minimize nephrotoxic dye load), requiring deformation estimation from the device silhouette and vessel edges alone. Moreover, AutoCAR's graph optimization—while ensuring topological correctness—requires 6 seconds per reconstruction, incompatible with continuous real-time feedback (<100 ms). The method also assumes temporally sparse key-frame selection, whereas device-induced deformation is continuous and spatially localized.

Complementary work on neural radiance fields (NeRF) for dynamic coronary arteries[cite:3] and other implicit 3D representations has shown promise for view synthesis and 4D reconstruction, but these methods optimize per-scene (requiring minutes of inference per case) and do not incorporate biomechanical constraints. The tension remains: purely geometric reconstruction achieves high fidelity but lacks physical plausibility, while physics-based simulation achieves realism but requires known boundary conditions (device forces, contact models) unavailable from 2D images alone.

### Autonomous Navigation and Interventional Robotics

An adjacent research direction—autonomous guidewire navigation—offers insights into device-vessel interaction modeling. Scarponi et al.[cite:7] developed a zero-shot reinforcement learning controller trained on only four synthetic bifurcation patterns, generalizing to unseen vascular anatomies with 95% success rate in reaching random targets. Their nearly shape-invariant observation space—encoding relative tangent vectors, curvature projections, and velocity rather than absolute coordinates—enables transfer learning across geometries. Crucially, their training environment simulates guidewire physics using the SOFA framework with Timoshenko beam theory and contact detection, validating accurate device-vessel mechanics at 90 fps. When tested on deformed coronary arteries (cardiac-induced motion), the controller maintained 90% success, demonstrating robustness to anatomical variations.

While Scarponi et al. focus on *control* (predicting optimal device rotations) rather than *estimation* (inferring vessel deformation), their physics simulation infrastructure and observation space design offer transferable principles. Their work highlights the importance of sparse, semantically meaningful representations (curvature, tangent directions) over dense voxel grids for computational efficiency and generalization. However, their method assumes the vascular geometry is known (from pre-operative imaging), using shape-sensing fiber Bragg gratings (FBG) to track the 3D guidewire tip shape. In contrast, our problem requires *jointly* inferring both vessel deformation and device configuration from 2D projections alone.

---

## The Gap: Real-Time, Physics-Constrained Deformation from Contrast-Free Fluoroscopy

Synthesizing the literature reveals a critical unmet need: **no existing method simultaneously achieves real-time inference (<100 ms), physics-based anatomical plausibility, and applicability to contrast-free fluoroscopy in endovascular procedures.** Specifically:

1. **Yang et al.[cite:8]** track instrument tips with respiratory compensation but do not estimate vessel wall deformation or enforce biomechanical constraints.

2. **Amiri-Hezaveh et al.[cite:5] and Harper et al.[cite:6]** incorporate rigorous continuum mechanics but require volumetric imaging inputs (MRI, CT, depth sensors) incompatible with intraoperative fluoroscopy, and their computational costs preclude real-time deployment.

3. **Zhu et al.[cite:9]** reconstruct static 3D vasculature from sparse views but rely on contrast-enhanced angiography, graph optimization (slow), and do not model device-induced deformation.

4. **Lecotme et al.** (as cited in our preliminary work) estimate deformation from fluoroscopy using fully convolutional networks but achieve limited out-of-plane accuracy and lack physics constraints, leading to anatomically implausible predictions in validation.

5. **Scarponi et al.[cite:7]** demonstrate robust physics simulation for guidewire interaction but assume known vascular geometry, addressing navigation control rather than deformation estimation.

The gap persists because these approaches emerge from distinct research communities—interventional radiology (focusing on clinical workflow), medical image registration (focusing on geometric accuracy), and biomechanics simulation (focusing on physical realism)—with limited cross-pollination. Device-induced vascular deformation occupies an intersection requiring:

- **Computational efficiency:** Sparse parameterization (not dense voxel/mesh fields) to enable <100 ms inference  
- **Physical plausibility:** Strain energy regularization to prevent impossible deformations  
- **Projection-based inference:** Differentiable rendering to optimize from 2D fluoroscopy  
- **Generalization:** Training on synthetic data with sufficient domain randomization  

Moreover, the unique geometric and mechanical properties of vascular structures—high aspect ratios (length >> diameter), bending-dominated deformation, and topological constraints (centerline connectivity)—distinguish them from bulk soft tissues (liver, brain) where existing PINN methods have been validated. Vessels behave more like slender beams than elastic solids, requiring curvature-penalized energy formulations (e.g., Kirchhoff-Love thin shells, Euler-Bernoulli beams) rather than volumetric strain measures.

Clinically, this gap manifests as:

- **Safety risks:** Inability to detect vessel over-deformation in real-time increases perforation/dissection likelihood  
- **Cognitive overload:** Surgeons mentally reconstruct 3D anatomy from ambiguous 2D projections, delaying decisions  
- **Radiation exposure:** Repeated contrast injections and prolonged fluoroscopy compensate for uncertainty  
- **Limited automation:** Robotic assistance systems lack accurate intraoperative anatomical models for autonomous navigation  

---

## Our Contribution: Sparse Control Point Physics-Informed Deformation Estimation

Our work addresses this gap by introducing a **sparse control point radial basis function (RBF) framework with strain energy regularization** for real-time estimation of guidewire-induced vascular deformation from monoplane fluoroscopy. Our method extends and reconciles three bodies of work:

1. **Extending Yang et al.[cite:8]:** While Yang tracks *instrument position* within vessels, we estimate *vessel wall deformation* caused by device forces. Our RBF parameterization enables localized, high-curvature deformations at contact points, consistent with beam bending mechanics, complementing their respiratory motion modeling.

2. **Applying Amiri-Hezaveh[cite:5] and Harper[cite:6] to vascular structures:** We adapt PINN-based deformation modeling specifically to tubular, high-curvature vasculature by: (a) replacing volumetric FEM with centerline-based RBF interpolation, (b) incorporating bending and stretching energy penalties derived from beam theory, and (c) enabling training from 2D projections via differentiable rendering.

3. **Complementing Zhu et al.[cite:9]:** AutoCAR provides static 3D anatomy; our method estimates *dynamic deformation* during device manipulation. Together, they could enable a hybrid pipeline: AutoCAR reconstructs initial geometry from biplane angiography, then our method continuously updates deformation from monoplane fluoroscopy.

### Technical Novelty

Our framework introduces three key innovations:

**1. Sparse Control Point Parameterization**  
Rather than predicting dense displacement fields (computationally prohibitive) or relying on iterative optimization (too slow), we represent deformation via *N* sparse control points along the vascular centerline. A ResNet encoder processes the difference between the current 2D fluoroscopy image and the projected pre-operative model, predicting 3D displacement vectors (Δp ∈ ℝ^(N×3)) for each control point. Global vessel deformation is then interpolated via thin-plate spline RBF:

**d(x) = Σᵢ wᵢ · φ(||x - cᵢ||)**

where **cᵢ** are control point locations, **wᵢ** are predicted displacements, and **φ(r) = r² log(r)** enforces C² continuity. This parameterization reduces degrees of freedom from ~10⁵ (dense mesh vertices) to ~10¹ (sparse control points), enabling real-time inference while analytically guaranteeing smooth, continuous vessel shapes—the network *cannot* generate broken or kinked vessels.

**2. Differentiable Physics-Based Regularization**  
Our compound loss function balances photometric fidelity with biomechanical plausibility:

**ℒ = ℒ_photo + λ_bend · ℒ_bend + λ_stretch · ℒ_stretch**

- **ℒ_photo:** L2 pixel error between rendered projection and fluoroscopy  
- **ℒ_bend:** Bending energy penalty ∫ ||κ(s)||² ds (penalizes high curvature)  
- **ℒ_stretch:** Stretching energy penalty ∫ (ε_axial)² ds (penalizes length changes)  

where **κ(s)** is the local curvature (second derivative of centerline) and **ε_axial** is axial strain (first derivative). The entire pipeline—ResNet encoder → RBF deformation → differentiable X-ray projection—is end-to-end trainable, backpropagating pixel-level errors to 3D force vectors while respecting vascular mechanical limits.

Unlike unconstrained CNNs (which may predict impossible shapes), our physics-informed loss constrains the solution manifold to biomechanically feasible deformations. Unlike iterative FEM solvers, our feed-forward neural network amortizes optimization cost during training, achieving <100 ms inference.

**3. Synthetic Training with Domain Randomization**  
Training data is generated by: (a) extracting vascular centerlines from CTA volumes, (b) applying random RBF deformations driven by simulated guidewire forces, (c) rendering synthetic fluoroscopy via differentiable ray tracing with randomized C-arm poses, contrast levels, and noise. This approach mirrors AutoCAR's[cite:9] domain adaptation strategy but tailored to deformation scenarios. Critically, ground-truth deformation parameters (control point displacements) are known by construction, enabling supervised training without manual annotation.

### Expected Validation and Impact

We will validate our method on: (1) synthetic data with known ground-truth deformations (quantifying reprojection error and deformation recovery), (2) real endovascular thrombectomy videos acquired at University of Pittsburgh, comparing estimated deformations against post-procedure CTA when available, and (3) ablation studies isolating the contribution of physics-based regularization versus purely data-driven baselines (e.g., Lecotme et al.). Success criteria include: reprojection error <2 mm (comparable to AutoCAR's static reconstruction), inference time <100 ms (enabling 10 Hz surgical feedback), and qualitative assessment by interventionalists of deformation plausibility.

If successful, this work will enable three clinical advances:

1. **Intraoperative safety monitoring:** Real-time alerts when vessel strain exceeds safe thresholds, preventing complications  
2. **Augmented fluoroscopy:** Overlaying deformed 3D models onto live 2D images, improving spatial awareness  
3. **Robotic guidance integration:** Providing accurate anatomical models for semi-autonomous navigation systems (e.g., Scarponi et al.[cite:7])  

More broadly, our sparse control point approach and differentiable physics pipeline provide a generalizable framework for other interventional scenarios—cardiac catheterization, peripheral artery stenting, endovascular aneurysm repair—where device-tissue interaction drives anatomical change. By bridging the gap between real-time intraoperative imaging and physics-based deformation modeling, this work supports the broader vision of intelligent surgical assistance systems that augment human expertise without replacing it.

---

## References

[cite:1] (To be added: foundational vascular biomechanics paper, e.g., Humphrey's "Cardiovascular Solid Mechanics" or similar)

[cite:3] NeRF-CA: Dynamic Reconstruction of X-Ray Coronary Angiography With Extremely Sparse-Views. (Provided PDF - full citation to be formatted per journal style)

[cite:4] (To be added: neural radiance fields or implicit representations paper relevant to 4D reconstruction)

[cite:5] Amiri-Hezaveh A, Tan S, Deng Q, Umulis D, Cunniff L, Weickenmeier J, Buganza Tepole A. A Physics-Informed Deep Learning Deformable Medical Image Registration Method Based on Neural ODEs. *International Journal of Computer Vision*. 2025;133(6):6374-6399.

[cite:6] Harper DM, Chen LJ, McKay RT, Nguyen SL, Fontaine MA. Physics-Informed Neural Networks for Real-Time Deformation-Aware AR Surgical Tracking. *bioRxiv*. 2025. doi:10.1101/2025.09.23.678071.

[cite:7] Scarponi V, Duprez M, Nageotte F, Cotin S. A zero-shot reinforcement learning strategy for autonomous guidewire navigation. *International Journal of Computer Assisted Radiology and Surgery*. 2024;19(11):1185-1192.

[cite:8] Yang S, Xiao D, Geng H, Ai D, Fan J, Fu T, Song H, Duan F, Yang J. Real-Time 3D Instrument Tip Tracking Using 2D X-Ray Fluoroscopy With Vessel Deformation Correction Under Free Breathing. *IEEE Transactions on Biomedical Engineering*. 2025;72(4):1422-1436.

[cite:9] Zhu Y, et al. Sparse and transferable 3D dynamic vascular reconstruction for instantaneous diagnosis. *Nature Machine Intelligence*. 2025;7(5):730-742.

*Additional references to be formatted:*
- Lecotme et al. (FCN for deformation estimation - full citation needed)
- Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks (foundational PINN paper)
- Classical 2D-3D registration papers (Miao et al., Schaffert et al.)
- RBF deformation methods
- Differentiable rendering frameworks

---

## Notes on Revision

This draft provides:
1. **Status quo synthesis** across three research domains (tracking, physics-informed ML, reconstruction)
2. **Gap articulation** with specific technical and clinical dimensions
3. **Positioning of your contribution** as extending/complementing/contradicting prior work
4. **Embedded evidence** of which work influences your design (Yang's tracking, Amiri-Hezaveh's PINNs, AutoCAR's domain adaptation)

**Next steps for revision:**
- Add 3-5 more foundational citations (vascular biomechanics, classical registration, RBF theory, differentiable rendering)
- Expand clinical motivation in Introduction with statistics on complication rates / procedural times
- Add 1-2 paragraphs on related work in autonomous catheterization (if relevant to your framing)
- Refine transitions between subsections for smoother flow
- Adjust technical depth based on audience (more medical context for clinical journals, more algorithmic detail for MICCAI/IPMI venues)
- Format references according to target journal style (IEEE, Nature, SPIE, etc.)

The ~2,800 word length is appropriate for an IMRD introduction or a standalone literature review for coursework. For journal submission, this would be condensed to ~1,200-1,500 words in the Introduction section, with fuller treatment of methods moved to related work sidebars or supplementary material.
