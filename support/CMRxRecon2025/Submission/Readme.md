# Submission Guidelines

Please ensure to maintain the current directory structure and save the data into the corresponding subfolders as specified.

Name your zip file as `Submission.zip`.


### Directory Structure
For each modality, follow this structure:
`{Task}/MultiCoil/{Modality}/ValidationSet/{UnderSample_Task}/{Center}/{Vendor}/PXXX/{ModalityFileName}_kus_{SamplePatternName}{R}.mat`
- Note:`{Task}:TaskR1,TaskR2,TaskS1,TaskS2`
- To see the full directory layout, refer to `Format_TaskR1` and `Format_TaskR2`.

### Modalities
- **Modalities:** 'BlackBlood', 'Cine', 'Flow2d', 'LGE','Mapping', 'Perfusion', 'T1rho', 'T1w', 'T2w'
  - Note: For BlackBlood, T1w, T2w the data dimensions are `[nx, ny, nc, nz]`, which differ from the other modalities.

### Processing Instructions
All data should be coil-combined and cropped using the MATLAB script [`mainRun4Ranking2025.m`](../CMRxReconDemo/run4Ranking.m), available in the `CMRxReconDemo` subdirectory.

