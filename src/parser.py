from pathlib import Path
import openfoamparser_mai as Ofpp
import pandas as pd
import numpy as np
from typing import Union
import os
import re

def surface_to_dataframe(surface, u_path):

    ### surface and pressure

    X = np.zeros(len(surface[0]['Item2']))
    Y = np.zeros(len(surface[0]['Item2']))
    Z = np.zeros(len(surface[0]['Item2']))
    P = np.zeros(len(surface[0]['Item2']))
    Norm = np.zeros((len(surface[0]['Item2']),3))
    Face = np.zeros(len(surface[0]['Item2'])).astype('str')
    U = np.ones(len(surface[0]['Item2']))

    for index, dicts in zip(range(len(surface[0]['Item2'])), surface[0]['Item2']):
        X[index] = dicts['CentrePosition']['X']
        Y[index] = dicts['CentrePosition']['Y']
        Z[index] = dicts['CentrePosition']['Z']
        Norm[index]=dicts['normal']
        Face[index]=str(dicts['face'])
        P[index] = dicts['PressureValue']

    ### initial conditions

    with open(u_path, 'r') as file:
        lines = file.readlines()  

    target_line = lines[11] 

    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', target_line)

    if len(numbers) > 0:
        first_number_in_brackets = float(numbers[0])
    else:
        print("error")

    U = U * first_number_in_brackets
    print(Face[:10], Face.shape)
    df = pd.DataFrame(np.array([X, Y, Z, P, U]).T)
    df.columns = ['X', 'Y', 'Z', 'P','Ux']
    df[['Normalx','Normaly','Normalz']] = Norm
    df['Face'] = Face

    # df.to_parquet(f'./{name}.parquet', engine='pyarrow')

    return df 

def _face_center_position(points: list, mesh: Ofpp.FoamMesh) -> list:
    vertecis = [mesh.points[p] for p in points]
    vertecis = np.array(vertecis)
    return points,vertecis.mean(axis=0)

def pressure_field_on_surface(solver_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 surface_name: str = 'Surface') -> None:
    
    mesh_bin = Ofpp.FoamMesh(solver_path )

    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
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

        body_faces = []
        for f, b in f_b_set:
            try:
                face, position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                vec_1 = mesh_bin.points[face[0]]
                vec_2 = mesh_bin.points[face[1]]
                vec_3 = mesh_bin.points[face[2]]
                
                new_vec_first = [vec_2[0] - vec_1[0],vec_2[1] - vec_1[1],vec_2[2] - vec_1[2]]
                new_vec_sec = [vec_2[0] - vec_3[0],vec_2[1] - vec_3[1],vec_2[2] - vec_3[2]]

                normal = np.cross(new_vec_first,new_vec_sec)
                normal = normal/np.linalg.norm(normal)

                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'normal': normal,
                    'face': face,
                    'CentrePosition': {'X': position[0], 'Y': position[1], 'Z': position[2]},
                    'PressureValue': p[b]
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,'Item2': body_faces}) 
        
        return surfaces

def parser_pipeline(foam_path) -> pd.DataFrame:
    df = pd.DataFrame() 

    end_time = re.search("(^.+/)(.+?/)([0-9.]+)$", foam_path)[3]
    new_path = re.search("(^.+/)(.+?/)([0-9.]+)$", foam_path)[2]
    original_path = re.search("(^.+/)(.+?/)([0-9.]+)$", foam_path)[1]

    PATH_TO_CASE = original_path + new_path
    END_TIME = end_time
    print(PATH_TO_CASE)

    base_path = Path(PATH_TO_CASE)
    time_path = base_path / Path(END_TIME)
    p_path = time_path / Path('p')
    p = Ofpp.parse_internal_field(p_path)
    # u = Ofpp.parse_internal_field(PATH_TO_CASE / Path('u'))
    
    surface = pressure_field_on_surface(base_path, p)

    u_path = PATH_TO_CASE + '0.orig/include/initialConditions'

    df = surface_to_dataframe(surface, u_path)
    
    # На случай, если передаются тренировочные данные
    if 'P' in df.columns:
        df = df.drop(columns=['P'])
    if 'Face' in df.columns:
        df = df.drop(columns=['Face'])
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