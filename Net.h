//
// Created by marek on 10/25/2021.
//

#ifndef SRC_NET_H
#define SRC_NET_H


class Net {
public:
    Net(topology);
//    input shape 28*28 = 784 vals
//tiez by sa dalo pouzit std::vector<double>
    void forward(double input[]);
    void backward(double target[]);
};


#endif //SRC_NET_H
