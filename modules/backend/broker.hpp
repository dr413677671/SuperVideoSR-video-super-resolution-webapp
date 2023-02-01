#include <libavcodec/avcodec.h>
#include<Queue>
#include<signal.h>
#include<mutex>

class PacketQueue()
{
public:
    void lockRead()
    {
        lock_guard(RLock);
        if (++readCnt == 1) {
            WLock.lock(); 
        }
    }

    void unlockRead()
    {
        lock_guard(RLock);
        if (--readCnt == 0) {
            WLock.unlock(); 
        }
    }

    void lockWrtite()
    {
       WLock.lock();
    }

    void unlockWrtite()
    {
       WLock.unlock();
    }

    uint32_t push(AVPacket* packet)
    {
        lockWrtite();
        m_queue.push(packet);
        unlockWrtite();
    }

    uint32_t pop(AVPacket* packet)
    {
        lockRead();
        m_queue.pop(packet);
        unlockRead();
    }

    std::queue<AVPacket> m_queue;
    std::mutex RLock;
    std::mutex WLock;
    int readCnt;
}