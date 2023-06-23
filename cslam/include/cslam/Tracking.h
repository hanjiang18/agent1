/**
* This file is part of CCM-SLAM.
*
* Copyright (C): Patrik Schmuck <pschmuck at ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/patriksc/CCM-SLAM>
*
* CCM-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CCM-SLAM is based in the monocular version of ORB-SLAM2 by Raúl Mur-Artal.
* CCM-SLAM partially re-uses modules of ORB-SLAM2 in modified or unmodified condition.
* For more information see <https://github.com/raulmur/ORB_SLAM2>.
*
* CCM-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CCM-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef CSLAM_TRACKING_H_
#define CSLAM_TRACKING_H_

//C++
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <mutex>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

//CSLAM
#include <cslam/config.h>
#include <cslam/estd.h>
#include <cslam/Converter.h>
#include <cslam/ORBextractor.h>
#include <cslam/ORBVocabulary.h>
#include <cslam/ORBmatcher.h>
#include <cslam/Frame.h>
#include <cslam/MapPoint.h>
#include <cslam/Map.h>
#include <cslam/KeyFrame.h>
#include <cslam/Mapping.h>
#include <cslam/PnPSolver.h>
#include <cslam/Optimizer.h>
#include <cslam/Viewer.h>
#include <cslam/Initializer.h>
#include <cslam/CentralControl.h>
#include <cslam/Viewer.h>
#include <cslam/Communicator.h>
#include <cslam/ClientHandler.h>
#include <cslam/client/ClientSystem.h>

using namespace std;
using namespace estd;

namespace cslam{

//forward decs
class Viewer;
class Initializer;
class Frame;
class LocalMapping;
class KeyFrameDatabase;
class CentralControl;
class KeyFrame;
class ORBmatcher;
class ClientHandler;
class ClientSystem;
//------------

class Tracking : public boost::enable_shared_from_this<Tracking>
{
public:
    typedef boost::shared_ptr<Tracking> trackptr;
    typedef boost::shared_ptr<Viewer> viewptr;
    typedef boost::shared_ptr<CentralControl> ccptr;
    typedef boost::shared_ptr<Map> mapptr;
    typedef boost::shared_ptr<KeyFrameDatabase> dbptr;
    typedef boost::shared_ptr<Communicator> commptr;
    typedef boost::shared_ptr<LocalMapping> mappingptr;
    typedef boost::shared_ptr<KeyFrame> kfptr;
    typedef boost::shared_ptr<MapPoint> mpptr;
    typedef boost::shared_ptr<Initializer> initptr;
    typedef boost::shared_ptr<Frame> frameptr;
    typedef boost::shared_ptr<ClientSystem> csptr;

    

public:
    Tracking(ccptr pCC, vocptr pVoc, viewptr pFrameViewer, mapptr pMap,
             dbptr pKFDB, const string &strCamPath, size_t ClientId);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

    // Pointer Setters
    void SetLocalMapper(mappingptr pLocalMapper) {mpLocalMapper = pLocalMapper;}
    void SetCommunicator(commptr pComm) {mpComm = pComm;}

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    int mSensor;

   

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
    frameptr mCurrentFrame;
    cv::Mat mImGray;

     //myadd
    cv::Mat mImRGB;
    cv::Mat imDepth ;
;// adding mImDepth member to realize pointcloud view

    kfptr GetReferenceKF();
    std::vector<mpptr> GetLocalMPs(){return mvpLocalMapPoints;}
    vector<int> GetValue(const cv::Mat &picture);
    bool Relocalization();

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    frameptr mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<kfptr> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    //my add
   idpair mnLastRelocFrameId;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for monocular
    void MonocularInitialization();

    // Map initialization for stereo and RGB-D
    /** @brief 在双目输入和RGBD输入时所做的初始化,主要是产生初始地图 */
    void StereoInitialization();
    
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    //infrastructure
    size_t mClientId;
    vocptr mpORBVocabulary;
    ccptr mpCC;
    viewptr mpViewer;
    mapptr mpMap;
    commptr mpComm;
    dbptr mpKeyFrameDB;

    //Other Thread Pointers
    mappingptr mpLocalMapper;

    //ORB
    extractorptr mpORBextractor;
    extractorptr mpIniORBextractor;

    // Initalization (only for monocular)
    initptr mpInitializer;

    //Local Map
    kfptr mpReferenceKF;
    std::vector<kfptr> mvpLocalKeyFrames;
    std::vector<mpptr> mvpLocalMapPoints;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;

    //New KeyFrame rules (according to fps)
    // 新建关键帧和重定位中用来判断最小最大时间间隔，和帧率有关
    int mMinFrames;
    int mMaxFrames;

    ///相机的基线长度 * 相机的焦距
    float mbf;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    ///用于区分远点和近点的阈值. 近点认为可信度比较高;远点则要求在两个关键帧中得到匹配
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    ///深度缩放因子,链接深度值和具体深度值的参数.只对RGBD输入有效
    float mDepthMapFactor;

    ///临时的地图点,用于提高双目和RGBD摄像头的帧间效果,用完之后就扔了
    list<mpptr> mlpTemporalPoints;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    kfptr mpLastKeyFrame;
    frameptr mLastFrame;
    idpair mLastKeyFrameId;
    idpair mLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;


};

} //end namespace

#endif
