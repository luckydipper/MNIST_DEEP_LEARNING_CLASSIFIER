#include "parser/mnists_parser.hpp"

vector< vector<unsigned char> > getInputImgs(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<unsigned char> > images;
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + path);
    
    // ubyte 파일 포맷에 따라 첫 16바이트는 헤더 정보를 포함
    file.seekg(16);

    while (true) {
        std::vector<unsigned char> image(28 * 28);
        if (!file.read(reinterpret_cast<char*>(image.data()), 28 * 28))
            break; // 파일의 끝에 도달하면 읽기 중단
        images.push_back(image);
    }

    file.close();
    return images;
}


std::vector<int> getLabels(const std::string& path) {
    // 파일을 바이너리 모드로 열기
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    // 파일 헤더를 읽어서 레이블의 수를 확인
    int magic_number = 0;
    int num_labels = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number); // 바이너리 데이터의 엔디안을 변경
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = __builtin_bswap32(num_labels); // 바이너리 데이터의 엔디안을 변경

    // 레이블 데이터를 읽어서 벡터에 저장
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

