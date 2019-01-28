/*
 * converter.cpp: Graph converter (Edge list <--> binary)
 * (c) Mohammad Mofrad, 2018
 * (e) m.hasanzadeh.mofrad@gmail.com 
 * Standalone compile commnad:
 * g++ -o converter converter.cpp  -std=c++14 -DNDEBUG -O3 -flto -fwhole-program -march=native
 */
 
#include <iostream>
#include <cstdlib> 
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>

int main(int argc, char **argv) {
    /* 0 0: in.bin/txt   --> out.bin
       0 1: in.bin/txt   --> out.w.bin
       1 0: in.w.bin/txt --> out.bin
       1 1: in.w.bin/txt --> out.w.bin
    */
	if (argc != 7 and argc != 8) {
        std::cout << "Usage: " << argv[0] << " <filepath_in> <filepath_in is [0(txt)|1(bin)]> <filepath_in is weighted [0|1]>  <filepath_out> <Want filepath_out be [0(txt)|1(bin)]> <Want filepath_out be weighted [0|1]> [<offsetted by ?>]"  << std::endl;
    	exit(1);
	}

	std::string filepath_in  = argv[1];
    bool in_is_binary        = (atoi(argv[2]) == 1) ? true : false;
    bool in_is_weighted      = (atoi(argv[3]) == 1) ? true : false;
    std::string filepath_out = argv[4];
    bool out_be_binary       = (atoi(argv[5]) == 1) ? true : false;
    bool out_be_weighted     = (atoi(argv[6]) == 1) ? true : false;
    uint32_t displacement    = (argc == 8) ? (atoi(argv[7])) : 0;
    
    std::ifstream fin;
    if(in_is_binary)
        fin.open(filepath_in.c_str(), std::ios_base::binary);
    else
        fin.open(filepath_in.c_str(),   std::ios_base::in);
    
    if(!fin.is_open()) {
        fprintf(stderr, "Unable to open input file\n");
        exit(1); 
    }
    
    std::ofstream fout;
    if(out_be_binary)
        fout.open(filepath_out.c_str(), std::ios_base::binary);
    else
        fout.open(filepath_out.c_str(), std::ios_base::out);
    
    // Consume comments identified by '#' and '%' characters
    // Linebased read/write of pairs
    // Read pairs of (char i, char j) from filepath_in
    // Write bianry pairs of (uint32_t i, uint32_t j) to filepath_out
    uint32_t num_comments = 0;
    uint64_t num_edges = 0;
    uint32_t num_vertices = 0;
    bool is_comment = false;
    uint32_t i, j, w;
    if(in_is_binary) {
        // Obtain filesize
        uint64_t filesize, offset = 0;
        fin.seekg (0, std::ios_base::end);
        filesize = (uint64_t) fin.tellg();
        fin.seekg(0, std::ios_base::beg);

        while (offset < filesize) {
            fin.read(reinterpret_cast<char*>(&i), sizeof(uint32_t));
            i = i + displacement;
            offset += sizeof(uint32_t);
            fin.read(reinterpret_cast<char*>(&j), sizeof(uint32_t));
            j = j + displacement;
            offset += sizeof(uint32_t);
            
            if(in_is_weighted) {
                fin.read(reinterpret_cast<char*>(&w), sizeof(uint32_t));
                offset += sizeof(uint32_t);
            }
            else
                w = 1 + (std::rand() % (127 + 1));
            
            if(out_be_binary) {
                fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
                fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
            }
            else
                fout << i << " " << j;
            
            if(out_be_weighted) {
                if(out_be_binary)
                    fout.write(reinterpret_cast<const char*>(&w), sizeof(uint32_t));
                else
                    fout << " " << w;
            }
            
            if(not out_be_binary)
                fout << "\n";
            
            //printf("%d %d %d\n", i, j, w);
            num_edges++;
            num_vertices = (num_vertices < i) ? i : num_vertices;
            num_vertices = (num_vertices < j) ? j : num_vertices;
        }
        fout.close();
        fin.close();
        assert(offset == filesize);
    }
    else
    {
        std::string line;
        std::istringstream iss;
        while (std::getline(fin, line)) {
            if ((line[0] == '#') || (line[0] == '%')) {
                
                if(!is_comment) {
                    is_comment = true;
                    std::cout << "########################################" << std::endl;
                }
                std::cout << line << std::endl; //NoOP
                num_comments++;
            } 
            else {
                iss.clear();
                iss.str(line);
                //std::cout << "i=" << i << "j=" << j << std::endl;
                if(in_is_weighted)
                    iss >> i >> j >> w;
                else {
                    w = 1 + (std::rand() % (127 + 1));
                    iss >> i >> j;
                }
                i = i + displacement;
                j = j + displacement;
                
                if(out_be_binary) {
                    fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
                    fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
                }
                else
                    fout << i << " " << j;
                
                if(out_be_weighted) {
                    if(out_be_binary)
                        fout.write(reinterpret_cast<const char*>(&w), sizeof(uint32_t));
                    else
                        fout << " " << w;
                }
                
                if(not out_be_binary)
                    fout << "\n";
                
                num_edges++;
                num_vertices = (num_vertices < i) ? i : num_vertices;
                num_vertices = (num_vertices < j) ? j : num_vertices;
            }
        }
        fout.close();
        fin.close();
    }
	std::cout << "########################################" << std::endl;
	std::cout << "Read/write stats:" << std::endl;
	std::cout << num_comments << " line comments" << std::endl;
    std::cout << num_vertices << " vertices (excluding zero)" << std::endl;
	std::cout << num_edges << " edges " << std::endl;
	std::cout << (num_comments + num_edges) << " number of lines" << std::endl;
	std::cout << "Verify using \"wc -l " << filepath_in << "\"" << std::endl;

	return(0);
}