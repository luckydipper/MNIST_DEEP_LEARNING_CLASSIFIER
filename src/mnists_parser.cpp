#include "mnists_parser.hpp"

vector< vector<unsigned char> > getData(string path){
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<unsigned char> > images;

    if (file.is_open()) {
        // ubyte 파일 포맷에 따라 첫 16바이트는 헤더 정보를 포함
        file.seekg(16);

        while (true) {
            std::vector<unsigned char> image(28 * 28);
            if (!file.read(reinterpret_cast<char*>(image.data()), 28 * 28)) {
                break; // 파일의 끝에 도달하면 읽기 중단
            }

            images.push_back(image);
        }

        file.close();
    } else {
        cout << "Unable to open file " << path << "\n";
    }   //cerr

    return images;
}