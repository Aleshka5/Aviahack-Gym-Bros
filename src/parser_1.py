from pathlib import Path
import openfoamparser_mai as Ofpp
import pandas as pd
import numpy as np
from typing import Union
import os
import re
import torch

def pressure_field_on_surface_to_dataframe(solver_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 surface_name: str = 'Surface',
                                 with_normals: bool = True):    
    mesh_bin = Ofpp.FoamMesh(solver_path )

    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = {'ParentElementID': [],
                'ParentFaceId': [],
                'norm_x': [],
                'norm_y': [],
                'norm_z': [],                
                'X': [], 
                'Y': [], 
                'Z': [],
                'PressureValue': []}    
    
    for _, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))
         
        # Define variables                      
        vectors_1_cpu = np.zeros((len(f_b_set),3))
        vectors_2_cpu = np.zeros((len(f_b_set),3))
        vectors_3_cpu = np.zeros((len(f_b_set),3))        
        i = 0 
        for f, b in f_b_set:        
            try:
                # face, position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                face = mesh_bin.faces[f]                
                vertices = [mesh_bin.points[p] for p in face]
                vertices = np.array(vertices)
                position = vertices.mean(axis=0)
                # ----------------------------------                

                if with_normals:
                    # Расчёт нормалей
                    vectors_1_cpu[i] = mesh_bin.points[face[0]]
                    vectors_2_cpu[i] = mesh_bin.points[face[1]]
                    vectors_3_cpu[i] = mesh_bin.points[face[2]]                                        

                    surfaces['ParentElementID'].append(b)
                    surfaces['ParentFaceId'].append(f)
                    surfaces['X'].append(position[0])
                    surfaces['Y'].append(position[1])
                    surfaces['Z'].append(position[2])
                    surfaces['PressureValue'].append(p[b])                        
                else:
                    surfaces['ParentElementID'].append(b)
                    surfaces['ParentFaceId'].append(f)
                    surfaces['X'].append(position[0])
                    surfaces['Y'].append(position[1])
                    surfaces['Z'].append(position[2])
                    surfaces['PressureValue'].append(p[b])                        
                # body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')        
            i += 1             
        
        # Normal GPU Calculate
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        vectors_1_gpu = torch.tensor(vectors_1_cpu,device=device)
        vectors_2_gpu = torch.tensor(vectors_2_cpu,device=device)
        vectors_3_gpu = torch.tensor(vectors_3_cpu,device=device)        
        
        new_vec_first = vectors_2_gpu - vectors_1_gpu
        new_vec_sec = vectors_2_gpu - vectors_3_gpu        
        # print(f'First vec {new_vec_first[:3]}')
        # print(f'Sec vec {new_vec_sec[:3]}')
        # Вычисление векторного произведения
        normal_gpu = torch.linalg.cross(new_vec_first, new_vec_sec)
        # print(f'Normal {normal_gpu[:3]},{normal_gpu.shape}')
        norma = torch.linalg.norm(normal_gpu,dim=1)
        # print(f'Norma {norma}, {norma.shape}')
        normal_gpu = normal_gpu.T / norma
        normal_cpu = normal_gpu.T.cpu()
        # print(f'Normal normalize {normal_cpu[:3]}')
        
        torch.cuda.synchronize()    
        
        surfaces['norm_x'] = normal_cpu[:,0].tolist()
        surfaces['norm_y'] = normal_cpu[:,1].tolist()
        surfaces['norm_z'] = normal_cpu[:,2].tolist()                      
        # for i in surfaces.keys():
        #     print(i,len(surfaces[i]))
        return pd.DataFrame(surfaces,columns=surfaces.keys())

def parser_pipeline(foam_path, is_train=False, with_normals=True) -> pd.DataFrame:
    df = pd.DataFrame() 

    end_time = re.search("(^.+/)(.+?/)([0-9.]+)$", foam_path)[3]
    new_path = re.search("(^.+/)(.+?/)([0-9.]+)$", foam_path)[2]
    original_path = re.search("(^.+/)(.+?/)([0-9.]+)$", foam_path)[1]

    PATH_TO_CASE = original_path + new_path
    END_TIME = end_time    

    base_path = Path(PATH_TO_CASE)
    time_path = base_path / Path(END_TIME)
    p_path = time_path / Path('p')
    p = Ofpp.parse_internal_field(p_path)    
    
    df = pressure_field_on_surface_to_dataframe(base_path, p, with_normals=with_normals)
    
    return df


if __name__ == '__main__':
    
# luna: '0.3M','0.5M','0.7M','0.9M','1.1M','1.3M','1.5M','1.7M','1.9M','2.0M'
#        150,    150,   150,   150,   40,    40,    40,    40,    40,    40
#        range(10)
# agard: '150.0','183.3','216.6','250.0','280.0','303.3','326.6','350.0'
#        [150,     150,    150,    150,    150,    150,    150,    150   ],
#        range(8)):

    # for mah, end, index in zip(['0.3M','0.5M','0.7M','0.9M','1.1M','1.3M','1.5M','1.7M','1.9M','2.0M'],
    #                            [150,    150,   150,   150,   40,    40,    40,    40,    40,    40],
    #                            range(10)):
    df = parser_pipeline(f'D:/хакатон/case_2_field_prediction-main/data_folder/crm/crm0.0/200')
    #print(index, df.head)
    df.to_parquet('crm0.parquet',engine='pyarrow')