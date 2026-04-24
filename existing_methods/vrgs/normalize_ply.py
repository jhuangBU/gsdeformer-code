import numpy as np
import struct
import argparse

def read_ply_header(file):
    header = []
    while True:
        line = file.readline().decode('utf-8').strip()
        header.append(line)
        if line == 'end_header':
            break
    return header

def write_ply_header(file, header):
    for line in header:
        file.write((line + '\n').encode('utf-8'))

def process_ply_file(input_path, output_path):
    # Open input file in binary mode
    with open(input_path, 'rb') as f:
        # Read and parse header
        header = read_ply_header(f)
        
        # Calculate number of properties and identify fea indices
        properties = [line for line in header if line.startswith('property')]
        property_count = len(properties)
        fea_indices = [i for i, prop in enumerate(properties) 
                      if 'fea_' in prop]
        
        # Calculate the size of each vertex (4 bytes per float)
        vertex_size = property_count * 4
        
        # Get number of vertices
        for line in header:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[2])
                break
        
        # Read all vertex data
        vertex_data = np.frombuffer(
            f.read(num_vertices * vertex_size), 
            dtype=np.float32
        ).reshape(num_vertices, property_count)
        
        # Create new header without fea properties
        new_header = [
            line for line in header 
            if not (line.startswith('property') and 'fea_' in line)
        ]
        
        # Remove fea columns from vertex data
        vertex_data = np.delete(vertex_data, fea_indices, axis=1)
        
        # Write to output file
        with open(output_path, 'wb') as out_f:
            # Write new header
            write_ply_header(out_f, new_header)
            
            # Write vertex data
            out_f.write(vertex_data.tobytes())

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process PLY files by removing feature columns')
    parser.add_argument('--input', type=str, help='Input PLY file path')
    parser.add_argument('--output', type=str, help='Output PLY file path')

    # Parse arguments
    args = parser.parse_args()

    # Process the PLY file
    print(f"Processing {args.input} to {args.output}...")
    process_ply_file(args.input, args.output)
    print("Done!")

if __name__ == "__main__":
    main()
