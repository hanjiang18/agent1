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


#include <cslam/Tracking.h>


namespace cslam {

Tracking::Tracking(ccptr pCC, vocptr pVoc, viewptr pFrameViewer, mapptr pMap, dbptr pKFDB, const string &strCamPath, size_t ClientId)
    : mState(NO_IMAGES_YET),mpCC(pCC),mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpInitializer(nullptr),
      mpViewer(pFrameViewer), mpMap(pMap), mLastRelocFrameId(make_pair(0,0)), mClientId(ClientId)
{
    cv::FileStorage fSettings(strCamPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf=fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    const int nFeatures = params::extractor::miNumFeatures;
    const float fScaleFactor = params::extractor::mfScaleFactor;
    const int nLevels = params::extractor::miNumLevels;
    const int iIniThFAST = params::extractor::miIniThFAST;
    const int iMinThFAST = params::extractor::miNumThFAST;

    mpORBextractor.reset(new ORBextractor(nFeatures,fScaleFactor,nLevels,iIniThFAST,iMinThFAST
                                          ));
    mpIniORBextractor.reset(new ORBextractor(2*nFeatures,fScaleFactor,nLevels,iIniThFAST,iMinThFAST
                                             ));
    if(!mpMap || !mpORBVocabulary || !mpKeyFrameDB || !mpCC)
    {
        cout << "\033[1;31m!!!!! ERROR !!!!!\033[0m " << __func__ << ": nullptr given"<< endl;
        if(!mpMap) cout << "mpMap == nullptr" << endl;
        if(!mpORBVocabulary) cout << "mpORBVocabulary == nullptr" << endl;
        if(!mpKeyFrameDB) cout << "mpKeyFrameDB == nullptr" << endl;
        if(!mpCC) cout << "mpCC == nullptr" << endl;
        throw estd::infrastructure_ex();
    }

    if(!mpViewer)
    {
        cout << "\033[1;31m!!!!! ERROR !!!!!\033[0m " << __func__ << ": nullptr given"<< endl;
        if(!mpViewer) cout << "mpViewer == nullptr" << endl;
        throw estd::infrastructure_ex();
    }

    mThDepth= mbf*(float)fSettings["ThDepth"]/fx;
    cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;

    mDepthMapFactor = fSettings["DepthMapFactor"];
    if(fabs(mDepthMapFactor)<1e-5)
        mDepthMapFactor=1;
    else
        mDepthMapFactor = 1.0f/mDepthMapFactor;
    
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame.reset(new Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mClientId));
    else
    {
        mCurrentFrame.reset(new Frame(mImGray,timestamp,mpORBextractor,mpORBVocabulary,mK,mDistCoef,mClientId));
    }

    Track();

    return mCurrentFrame->mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp){
    mImGray = im;
    mImRGB=im;
    imDepth = depthmap;

    // step 1：将RGB或RGBA图像转为灰度图像
    /*if(mImGray.channels()==3)
    {
        if(mbRGB){
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
           
            vector<int> res=GetValue(mImGray);
            cout<<res[1]<<" "<<res[0]<<endl;
            if (res[1]<1){
                if (res[0] > 0){
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过亮 +++ !!!\033[0m" <<res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
                 else {
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过暗 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
            }
        }
            
        else{
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            
            vector<int> res=GetValue(mImGray);
            if (res[1]> 2){
                if (res[0] > 0){
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过亮 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
                 else {
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过暗 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
            }
        }       
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB){
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            
            vector<int> res=GetValue(mImGray);
            if (res[1]> 2){
                if (res[0] > 0){
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过亮 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
                 else {
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过暗 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
            }
        }
            
        else{
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            
            vector<int> res=GetValue(mImGray);
            if (res[1]> 2){
                if (res[0] > 0){
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过亮 +++ !!!\033[0m" <<res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
                 else {
                    std::cout << "\033[1;33m!!! +++ 亮度异常 过暗 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
            }
        }
            
    }*/

auto collections = dong_enhance(mImRGB, 3, 100, 0.8, 0.5, false);
    mImRGB= collections.back().second;
     cvtColor(mImRGB,mImGray,CV_RGB2GRAY);

     vector<int> res=GetValue(mImGray);
     //cout<<res[1]<<" "<<res[0]<<endl;
            if (res[1]>2){
                if (res[0] > 0){
                    //std::cout << "\033[1;33m!!! +++ 亮度异常 过亮 +++ !!!\033[0m" <<res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
                 else {
                    //std::cout << "\033[1;33m!!! +++ 亮度异常 过暗 +++ !!!\033[0m" << res[0] << std::endl;
                    cv::Ptr<CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	                clahe->setClipLimit(4);
	                clahe->setTilesGridSize(cv::Size(10, 10));
	                clahe->apply(mImGray, mImGray);
                }
            }
    // step 2 ：将深度相机的disparity转为Depth , 也就是转换成为真正尺度下的深度
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(  //将图像转换成为另外一种数据类型,具有可选的数据大小缩放系数
            imDepth,            //输出图像
            CV_32F,             //输出图像的数据类型
            mDepthMapFactor);   //缩放系数

    // 步骤3：构造Frame
    mCurrentFrame.reset(new Frame(mImGray,imDepth,timestamp,mpORBextractor,mpORBVocabulary,mK,mDistCoef, mbf,
        mThDepth,mClientId));


    // 步骤4：跟踪
 
    Track();

    //返回当前帧的位姿
    //cout<<mCurrentFrame->mTcw.clone()<<endl;
    //cout<<mCurrentFrame->mId.first<<endl;
    return mCurrentFrame->mTcw.clone();
}

vector<int> Tracking::GetValue(const cv::Mat &picture){
    float sum = 0;
    float avg = 0;
    cv::Scalar scalar;
    int ls[256];
    int size = picture.rows * picture.cols;
    for (int i = 0; i < 256; i++)
        ls[i] = 0;
    for (int i = 0; i < picture.rows; i++)
    {
        for (int j = 0; j < picture.cols; j++)
        {
            //scalar = cvGet2D(gray, i, j);
            scalar = picture.at<uchar>(i, j);
            sum += (scalar.val[0] - 128);
            int x = (int)scalar.val[0];
            ls[x]++;
        }
    }
    avg = sum / size;
    float total = 0;
    float mean = 0;
    for (int i = 0; i < 256; i++)
    {
        total += abs(float(i - 128) - avg) * ls[i];
    }
    mean = total / size;
    float cast = abs(avg / mean);
    //cout<<"cast: "<<cast<<"             avg :"<<avg<<endl;
    return {avg,cast};
     
 }

void Tracking::Track()
{
    //cout<<"state "<<mState<<endl;
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    while(!mpMap->LockMapUpdate()){
        usleep(params::timings::miLockSleep);
    }
    //Comm Mutex cannot be acquired here. In case of wrong initialization, there is mutual dependency in the call of reset()
    if(mState==NOT_INITIALIZED)
    {
            //单目初始化
        //MonocularInitialization();
        //RGBD相机的初始化
        StereoInitialization();
        
         if(params::vis::mbActive)
            mpViewer->UpdateAndDrawFrame();
        else
        {
            cout << "\033[1;35m!!! +++ Tracking: Init +++ !!!\033[0m" << endl;
            cout<<"wrong ----"<<endl;
        }

        if(mState!=OK)
        {
            mpMap->UnLockMapUpdate();
            return;
        }
    }
    else
    {
        //cout<<"here"<<endl;
        // Get Communicator Mutex -> Comm cannot publish. Assure no publishing whilst changing data
        if(params::sys::mbStrictLock) while(!mpCC->LockTracking()){
            usleep(params::timings::miLockSleep);
        }

        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(mState==OK)
        {
            // Local Mapping might have changed some MapPoints tracked in last frame
            
            CheckReplacedInLastFrame();

            if(mVelocity.empty() || mCurrentFrame->mId.first<mLastRelocFrameId.first+2)
            {
                
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                
                bOK = TrackWithMotionModel();
    
                if(!bOK){
                    
                    bOK = TrackReferenceKeyFrame();
                }
            }
        }
        else
        {
            bOK = Relocalization();
            cout << "\033[1;35m!!! +++ Tracking:   \033[0m" <<bOK<< endl;
            // cout << "\033[1;35m!!! +++ Tracking: Lost +++ !!!\033[0m" << endl;
            // bOK = false;
        }

        mCurrentFrame->mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(bOK) bOK = TrackLocalMap();

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        if(params::vis::mbActive) mpViewer->UpdateAndDrawFrame();

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame->mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame->GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame->GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame->mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();
            // Clean VO matches
            for(int i=0; i<mCurrentFrame->N; i++)
            {
                mpptr pMP = mCurrentFrame->mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame->mvbOutlier[i] = false;
                        mCurrentFrame->mvpMapPoints[i]=nullptr;
                    }
            }

            // Check if we need to insert a new keyframe
             if(NeedNewKeyFrame()){
                 CreateNewKeyFrame();
                }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame->N;i++)
            {
                if(mCurrentFrame->mvpMapPoints[i] && mCurrentFrame->mvbOutlier[i])
                    mCurrentFrame->mvpMapPoints[i]=nullptr;
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=params::tracking::miInitKFs)
            {
                //cout<<"mstat "<<mState<<endl;
                cout << "Track lost soon after initialisation, reseting..." << endl;
                if(params::sys::mbStrictLock) mpCC->UnLockTracking();
                mpMap->UnLockMapUpdate();
                mpCC->mpCH->Reset();

                return;
            }
        }

        if(!mCurrentFrame->mpReferenceKF)
            mCurrentFrame->mpReferenceKF = mpReferenceKF;

        mLastFrame.reset(new Frame(*mCurrentFrame));

        if(params::sys::mbStrictLock) mpCC->UnLockTracking();
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame->mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame->mTcw*mCurrentFrame->mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame->mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

    mpMap->UnLockMapUpdate();
    //cout<<"tracking : "<<mState<< endl;
}

void Tracking::MonocularInitialization()
{
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame->mvKeys.size()>100)
        {
            mInitialFrame.reset(new Frame(*mCurrentFrame));
            mLastFrame.reset(new Frame(*mCurrentFrame));
            mvbPrevMatched.resize(mCurrentFrame->mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame->mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame->mvKeysUn[i].pt;

            if(mpInitializer) mpInitializer = nullptr;

            mpInitializer.reset(new Initializer(*mCurrentFrame,1.0,200));

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame->mvKeys.size()<=100)
        {
            mpInitializer = nullptr;
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(*mInitialFrame,*mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            //delete mpInitializer;
            mpInitializer = nullptr;
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(*mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame->SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame->SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

 void Tracking::StereoInitialization(){
    if(mCurrentFrame->N>500){
         //mInitialFrame.reset(new Frame(*mCurrentFrame));
       
        // Set Frame pose to the origin
        // 设定初始位姿为单位旋转，0平移
        mCurrentFrame->SetPose(cv::Mat::eye(4,4,CV_32F));
        // Create KeyFrame
        // 将当前帧构造为初始关键帧
         kfptr pKFini {new KeyFrame(*mCurrentFrame,mpMap,mpKeyFrameDB,mpComm,eSystemState::CLIENT,-1)};
        // Insert KeyFrame in the map
        // KeyFrame中包含了地图、反过来地图中也包含了KeyFrame，相互包含
        // 在地图中添加该初始关键帧
        pKFini->getPCinfo(this->mImRGB,this->imDepth);
         mpMap->AddKeyFrame(pKFini);
         for(int i=0; i<mCurrentFrame->N;i++){
            float z = mCurrentFrame->mvDepth[i];
            if(z>0){
                cv::Mat x3D=mCurrentFrame->UnprojectStereo(i);
                //// 将3D点构造为MapPoint
                mpptr pNewMp {new MapPoint(x3D,pKFini,mpMap,mClientId,mpComm,eSystemState::CLIENT,-1)};
                // a.观测到该MapPoint的关键帧
                // b.该MapPoint的描述子
                // c.该MapPoint的平均观测方向和深度范围
                pNewMp->AddObservation(pKFini,i);
                pNewMp->ComputeDistinctiveDescriptors();
                pNewMp->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMp);
                pKFini->AddMapPoint(pNewMp,i);

                mCurrentFrame->mvpMapPoints[i]=pNewMp;
            }
         }
          cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;
          // 在局部地图中添加该初始关键帧
          mpLocalMapper->InsertKeyFrame(pKFini);
          //更新当前帧为上一帧
         mLastFrame.reset(new Frame(*mCurrentFrame));
        mLastKeyFrameId=mCurrentFrame->mId;
        mpLastKeyFrame=pKFini;

        mvpLocalKeyFrames.push_back(pKFini);

        //? 这个局部地图点竟然..不在mpLocalMapper中管理?
        // 我现在的想法是，这个点只是暂时被保存在了 Tracking 线程之中， 所以称之为 local 
        // 初始化之后，通过双目图像生成的地图点，都应该被认为是局部地图点
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF=pKFini;
        mCurrentFrame->mpReferenceKF=pKFini;
        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        
        mState=OK;
    }
 }
void Tracking::CreateInitialMapMonocular()
{
    // Get Communicator Mutex -> Comm cannot publish. Assure no publishing whilst changing data
    while(!mpCC->LockTracking()){
        usleep(params::timings::miLockSleep);
    }

    // Create KeyFrames
    kfptr pKFini{new KeyFrame(*mInitialFrame,mpMap,mpKeyFrameDB,mpComm,eSystemState::CLIENT,-1)};
    kfptr pKFcur{new KeyFrame(*mCurrentFrame,mpMap,mpKeyFrameDB,mpComm,eSystemState::CLIENT,-1)};

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        mpptr pMP{new MapPoint(worldPos,pKFcur,mpMap,mClientId,mpComm,eSystemState::CLIENT,-1)};

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame->mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame->mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemntClient(mpMap,mClientId,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        mpCC->UnLockTracking();
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w,false);

    // Scale points
    vector<mpptr> vpAllMapPoints = pKFini->GetMapPointMatches();

    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            mpptr pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth,false);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame->SetPose(pKFcur->GetPose());
    mLastKeyFrameId=mCurrentFrame->mId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame->mpReferenceKF = pKFcur;

    mLastFrame.reset(new Frame(*mCurrentFrame));

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    mpCC->UnLockTracking();
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame->N; i++)
    {
        mpptr pMP = mLastFrame->mvpMapPoints[i];

        if(pMP)
        {
            mpptr pRep = pMP->GetReplaced();
            if(pRep)
            {
                vector<mpptr>::iterator vit = std::find(mLastFrame->mvpMapPoints.begin(),mLastFrame->mvpMapPoints.end(),pRep);

                if(vit != mLastFrame->mvpMapPoints.end())
                {
                    int curId = vit - mLastFrame->mvpMapPoints.begin();

                    const cv::Mat dMP = pRep->GetDescriptor();

                    const cv::Mat &dF_curId = mLastFrame->mDescriptors.row(curId);
                    const cv::Mat &dF_i = mLastFrame->mDescriptors.row(i);

                    double dist_curId = ORBmatcher::DescriptorDistance(dMP,dF_curId);
                    double dist_i = ORBmatcher::DescriptorDistance(dMP,dF_i);

                    if(dist_i <= dist_curId)
                    {
                        mLastFrame->mvpMapPoints[curId] = nullptr;
                        mLastFrame->mvpMapPoints[i] = pRep;
                    }
                    else
                    {
                        //keep old id -- do nothing
                    }
                }
                else
                {
                    mLastFrame->mvpMapPoints[i] = pRep;
                }
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame->ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<mpptr> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,*mCurrentFrame,vpMapPointMatches);

    if(nmatches<params::tracking::miTrackWithRefKfInlierThresSearch)
        return false;

    mCurrentFrame->mvpMapPoints = vpMapPointMatches;
    mCurrentFrame->SetPose(mLastFrame->mTcw);

    Optimizer::PoseOptimizationClient(*mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(mCurrentFrame->mvbOutlier[i])
            {
                mpptr pMP = mCurrentFrame->mvpMapPoints[i];

                mCurrentFrame->mvpMapPoints[i]=nullptr;
                mCurrentFrame->mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mLastFrameSeen = mCurrentFrame->mId;
                nmatches--;
            }
            else if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=params::tracking::miTrackWithRefKfInlierThresOpt; //10
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    kfptr pRef = mLastFrame->mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame->SetPose(Tlr*pRef->GetPose());

     if(mLastKeyFrameId.first==mLastFrame->mId.first)
        return;
    //rgbd
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame->N);
    
    for(int i=0; i<mLastFrame->N;i++)
    {
        
        float z = mLastFrame->mvDepth[i];
        //cout<<"z : "<<z<<endl;
        if(z>0)
        {
            // vDepthIdx第一个元素是某个点的深度,第二个元素是对应的特征点id
            vDepthIdx.push_back(make_pair(z,i));
        }
    }
    if(vDepthIdx.empty())
        return;

    // 按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

     int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++){
        int id = vDepthIdx[j].second;

        bool bCreateNew = false;

        mpptr pMP=mLastFrame->mvpMapPoints[id];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)      
        {
            // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // Step 2.3：需要创建的点，包装为地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
            // 反投影到世界坐标系中
            cv::Mat x3D = mLastFrame->UnprojectStereo(id);

            mpptr pNewMP {new MapPoint(x3D,mLastFrame, mpMap, mClientId,mpComm,eSystemState::CLIENT,id)};
                 // 特征点id

            // 加入上一帧的地图点中
            mLastFrame->mvpMapPoints[id]=pNewMP; 

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            // 因为从近到远排序，记录其中不需要创建地图点的个数
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
    
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame->SetPose(mVelocity*mLastFrame->mTcw);

    fill(mCurrentFrame->mvpMapPoints.begin(),mCurrentFrame->mvpMapPoints.end(),nullptr);

    // Project points seen in previous frame
    int th;
    th=7;
    int nmatches = matcher.SearchByProjection(*mCurrentFrame,*mLastFrame,th);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame->mvpMapPoints.begin(),mCurrentFrame->mvpMapPoints.end(),nullptr);
        nmatches = matcher.SearchByProjection(*mCurrentFrame,*mLastFrame,2*th);
    }

    if(nmatches<params::tracking::miTrackWithMotionModelInlierThresSearch) //20
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimizationClient(*mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(mCurrentFrame->mvbOutlier[i])
            {
                mpptr pMP = mCurrentFrame->mvpMapPoints[i];

                mCurrentFrame->mvpMapPoints[i]=nullptr;
                mCurrentFrame->mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mLastFrameSeen = mCurrentFrame->mId;
                nmatches--;
            }
            else if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    //cout<<nmatchesMap<<endl;
    return nmatchesMap>=params::tracking::miTrackWithMotionModelInlierThresOpt; //10
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimizationClient(*mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(!mCurrentFrame->mvbOutlier[i])
            {
                mCurrentFrame->mvpMapPoints[i]->IncreaseFound();
                mnMatchesInliers++;
            }

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame->mId.first<mLastRelocFrameId.first+params::tracking::miMaxFrames && mnMatchesInliers<50)
    {
        return false;
    }

    if(mnMatchesInliers<params::tracking::miTrackLocalMapInlierThres) //30
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();
    //cout<<"nkfs: "<<nKFs<<endl;

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame->mId.first<mLastRelocFrameId.first+params::tracking::miMaxFrames && nKFs>params::tracking::miMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    int nNonTrackedClose = 0;  //双目或RGB-D中没有跟踪到的近点
    int nTrackedClose= 0;       //双目或RGB-D中成功跟踪的近点（三维点）

        for(int i =0; i<mCurrentFrame->N; i++)
        {
            // 深度值在有效范围内
            if(mCurrentFrame->mvDepth[i]>0 && mCurrentFrame->mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame->mvpMapPoints[i] && !mCurrentFrame->mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }

        bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);
        //cout<<"bNeedToInsertClose: "<<bNeedToInsertClose<<endl;
        float thRefRatio = 0.75f;

    // 关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
    if(nKFs<2)
        thRefRatio = 0.4f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame->mId.first>=mLastKeyFrameId.first+params::tracking::miMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame->mId.first>=mLastKeyFrameId.first+params::tracking::miMinFrames && bLocalMappingIdle);
    const bool c1c =(mnMatchesInliers<nRefMatches*0.25 ||       //当前帧和地图点匹配的数目非常少
                      bNeedToInsertClose) ; 
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio/*params::tracking::mfThRefRatio*/|| bNeedToInsertClose /*|| ratioMap<thMapRatio*/) && mnMatchesInliers>params::tracking::miMatchesInliersThres);
    //cout<<"here0"<<endl;
    if((c1a||c1b||c1c)&&c2)
    {
        //cout<<"her01"<<endl;
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            //cout<<"here2"<<mpLocalMapper->KeyframesInQueue()<<endl;
            if(mpLocalMapper->KeyframesInQueue()<3)
                    //队列中的关键帧数目不是很多,可以插入
                    return true;
                else
                    //队列中缓冲的关键帧数目太多,暂时不能插入
                    return false;
        }
        //cout<<"here3"<<endl;
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    cout<<"create new kf : "<<mCurrentFrame->mId.first<<endl;
    
    if(!mpLocalMapper->SetNotStop(true))
        return;

    kfptr pKF{new KeyFrame(*mCurrentFrame,mpMap,mpKeyFrameDB,mpComm,eSystemState::CLIENT,-1)};
///extra
    std::vector<mpptr> vpM = pKF->GetMapPointMatches();
    for(vector<mpptr>::const_iterator vit = vpM.begin();vit!=vpM.end();++vit)
    {
        mpptr pMPi = *vit;

        if(!pMPi)
            continue;

        if(pMPi->isBad())
            continue;

        if(pMPi->mId.second != mClientId)
        {
            pMPi->SetMultiUse();
        }
    }
//
    mpReferenceKF = pKF;
    mCurrentFrame->mpReferenceKF = pKF;

    //my add
    mCurrentFrame->UpdatePoseMatrices();
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame->N);

    for(int i=0; i<mCurrentFrame->N; i++)
        {
            float z = mCurrentFrame->mvDepth[i];
            if(z>0)
            {
                // 第一个元素是深度,第二个元素是对应的特征点的id
                vDepthIdx.push_back(make_pair(z,i));
            }
        }
    if(!vDepthIdx.empty())
        {
            // Step 3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // Step 3.3：从中找出不是地图点的生成临时地图点 
            // 处理的近点的个数
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
                mpptr pMP =mCurrentFrame->mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame->mvpMapPoints[i] = static_cast<mpptr>(NULL);
                }

                // 如果需要就新建地图点，这里的地图点不是临时的，是全局地图中新建地图点，用于跟踪
                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame->UnprojectStereo(i);
                    mpptr pNewMP {new MapPoint(x3D,pKF,mpMap,mClientId,mpComm,eSystemState::CLIENT,-1)};
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame->mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    // 因为从近到远排序，记录其中不需要创建地图点的个数
                    nPoints++;
                }

                // Step 3.4：停止新建地图点必须同时满足以下条件：
                // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
                // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    pKF->getPCinfo(this->mImRGB,this->imDepth);
    mpLocalMapper->InsertKeyFrame(pKF);
    //cout<<"to local Mapper"<<endl;
    
    mpLocalMapper->SetNotStop(false);

    mLastKeyFrameId = mCurrentFrame->mId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<mpptr>::iterator vit=mCurrentFrame->mvpMapPoints.begin(), vend=mCurrentFrame->mvpMapPoints.end(); vit!=vend; vit++)
    {
        mpptr pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = nullptr;
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mLastFrameSeen = mCurrentFrame->mId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    int seen = 0;
    int bad = 0;
    int notinfrustrum = 0;

    // Project points in frame and check its visibility
    for(vector<mpptr>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        mpptr pMP = *vit;
        if(pMP->mLastFrameSeen == mCurrentFrame->mId)
        {
            ++seen;
            continue;
        }
        if(pMP->isBad())
        {
            ++bad;
            continue;
        }
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame->isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        else ++notinfrustrum;
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame->mId.first<mLastRelocFrameId.first+2)
        {
            th=5;
        }

        matcher.SearchByProjection(*mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
//    UpdateLocalKeyFrames();
//    UpdateLocalPoints();

    mvpLocalMapPoints = mpMap->GetAllMapPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<kfptr>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        kfptr pKF = *itKF;
        const vector<mpptr> vpMPs = pKF->GetMapPointMatches();

        int empty = 0;

        for(vector<mpptr>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            mpptr pMP = *itMP;
            if(!pMP)
            {
                ++empty;
                continue;
            }
            if(pMP->mTrackReferenceForFrame==mCurrentFrame->mId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mTrackReferenceForFrame=mCurrentFrame->mId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<kfptr,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            mpptr pMP = mCurrentFrame->mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<kfptr,size_t> observations = pMP->GetObservations();
                for(map<kfptr,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame->mvpMapPoints[i]=nullptr;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    kfptr pKFmax= nullptr;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<kfptr,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        kfptr pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mTrackReferenceForFrame = mCurrentFrame->mId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<kfptr>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        kfptr pKF = *itKF;

        const vector<kfptr> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<kfptr>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            kfptr pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mTrackReferenceForFrame!=mCurrentFrame->mId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mTrackReferenceForFrame=mCurrentFrame->mId;
                    break;
                }
            }
        }

        const set<kfptr> spChilds = pKF->GetChilds();
        for(set<kfptr>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            kfptr pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mTrackReferenceForFrame!=mCurrentFrame->mId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mTrackReferenceForFrame=mCurrentFrame->mId;
                    break;
                }
            }
        }

        kfptr pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mTrackReferenceForFrame!=mCurrentFrame->mId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mTrackReferenceForFrame=mCurrentFrame->mId;
                break;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame->mpReferenceKF = mpReferenceKF;
    }
}

void Tracking::Reset()
{
    cout << "System Reseting" << endl;
    mpViewer->RequestReset();
    mpLocalMapper->RequestReset();
    mpKeyFrameDB->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;
    MapPoint::nNextId = 0;

    if(mpInitializer)
        mpInitializer = nullptr;

    mpComm->RequestReset();

    mpMap->clear();

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    cout << "Reseting Done..." << endl;
}

Tracking::kfptr Tracking::GetReferenceKF()
{
    //Works w/o mutex, since tracking is locked when this method is called by comm

    if(!mpCC->IsTrackingLocked())
        cout << "\033[1;33m!!! WARN !!!\033[0m " << __func__ << ":" << __LINE__ << " Tracking assumed to be locked " << endl;

    return mpReferenceKF;
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // Step 1：计算当前帧特征点的词袋向量
    mCurrentFrame->ComputeBoW();
    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2：用词袋找到与当前帧相似的候选关键帧
    vector<kfptr> vpCandidateKFs =mpKeyFrameDB->DetectRelocalizationCandidates(mCurrentFrame);
    cout<<"size : "<<vpCandidateKFs.size()<<endl;
    // 如果没有候选关键帧，则退出
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);
    //每个关键帧的解算器
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    //每个关键帧和当前帧中特征点的匹配关系
    vector<vector<mpptr> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);
    
    //放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    //有效的候选关键帧数目
    int nCandidates=0;

    // Step 3：遍历所有的候选关键帧，通过词袋进行快速匹配，用匹配结果初始化PnP Solver
    for(int i=0; i<nKFs; i++)
    {
       kfptr pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // 当前帧和候选关键帧用BoW进行快速匹配，匹配结果记录在vvpMapPointMatches，nmatches表示匹配的数目
            int nmatches = matcher.SearchByBoW(pKF,*mCurrentFrame,vvpMapPointMatches[i]);
            cout<<"nmatches : "<<nmatches<<endl;
            // 如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 如果匹配数目够用，用匹配结果初始化EPnPsolver
                // 为什么用EPnP? 因为计算复杂度低，精度高
                PnPsolver* pSolver = new PnPsolver(*mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(
                    0.99,   //用于计算RANSAC迭代次数理论值的概率
                    10,     //最小内点数, 但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
                    300,    //最大迭代次数
                    4,      //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
                    0.5,    //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                    5.991); // 自由度为2的卡方检验的阈值,程序中还会根据特征点所在的图层对这个阈值进行缩放
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 这里的 P4P RANSAC是Epnp，每次迭代需要4个点
    // 是否已经找到相匹配的关键帧的标志
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // Step 4: 通过一系列操作,直到找到能够匹配上的关键帧
    // 为什么搞这么复杂？答：是担心误闭环
    while(nCandidates>0 && !bMatch)
    {
        //遍历当前所有的候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            // 忽略放弃的
            if(vbDiscarded[i])
                continue;
    
            //内点标记
            vector<bool> vbInliers;     
            
            //内点数
            int nInliers;
            
            // 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
            bool bNoMore;

            // Step 4.1：通过EPnP算法估计姿态，迭代5次
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // bNoMore 为true 表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                //  Step 4.2：如果EPnP 计算出了位姿，对内点进行BA优化
                Tcw.copyTo(mCurrentFrame->mTcw);
                // EPnP 里RANSAC后的内点的集合
                set<mpptr> sFound;

                const int np = vbInliers.size();
                //遍历所有内点
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame->mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame->mvpMapPoints[j]=NULL;
                }

                // 只优化位姿,不优化地图点的坐标，返回的是内点的数量
                int nGood = Optimizer::PoseOptimizationClient(*mCurrentFrame);

                // 如果优化之后的内点数目不多，跳过了当前候选关键帧,但是却没有放弃当前帧的重定位
                if(nGood<10)
                    continue;

                // 删除外点对应的地图点
                for(int io =0; io<mCurrentFrame->N; io++)
                    if(mCurrentFrame->mvbOutlier[io])
                        mCurrentFrame->mvpMapPoints[io]=static_cast<mpptr>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 4.3：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if(nGood<50)
                {
                    // 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中, 生成新的匹配
                    int nadditional = matcher2.SearchByProjection(
                        *mCurrentFrame,          //当前帧
                        vpCandidateKFs[i],      //关键帧
                        sFound,                 //已经找到的地图点集合，不会用于PNP
                        10,                     //窗口阈值，会乘以金字塔尺度
                        100);                   //匹配的ORB描述子距离应该小于这个阈值

                    // 如果通过投影过程新增了比较多的匹配特征点对
                    if(nadditional+nGood>=50)
                    {
                        // 根据投影匹配的结果，再次采用3D-2D pnp BA优化位姿
                        nGood = Optimizer::PoseOptimizationClient(*mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // Step 4.4：如果BA后内点数还是比较少(<50)但是还不至于太少(>30)，可以挽救一下, 最后垂死挣扎 
                        // 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口
                        // 这里的位姿已经使用了更多的点进行了优化,应该更准，所以使用更小的窗口搜索
                        if(nGood>30 && nGood<50)
                        {
                            // 用更小窗口、更严格的描述子阈值，重新进行投影搜索匹配
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame->N; ip++)
                                if(mCurrentFrame->mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame->mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(
                                *mCurrentFrame,          //当前帧
                                vpCandidateKFs[i],      //候选的关键帧
                                sFound,                 //已经找到的地图点，不会用于PNP
                                3,                      //新的窗口阈值，会乘以金字塔尺度
                                64);                    //匹配的ORB描述子距离应该小于这个阈值

                            // Final optimization
                            // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimizationClient(*mCurrentFrame);
                                //更新地图点
                                for(int io =0; io<mCurrentFrame->N; io++)
                                    if(mCurrentFrame->mvbOutlier[io])
                                        mCurrentFrame->mvpMapPoints[io]=NULL;
                            }
                            //如果还是不能够满足就放弃了
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                // 如果对于当前的候选关键帧已经有足够的内点(50个)了,那么就认为重定位成功
                if(nGood>=50)
                {
                    bMatch = true;
                    // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
                    break;
                }
            }
        }//一直运行,知道已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
    }

    // 折腾了这么久还是没有匹配上，重定位失败
    if(!bMatch)
    {
        return false;
    }
    else
    {
        // 如果匹配上了,说明当前帧重定位成功了(当前帧已经有了自己的位姿)
        // 记录成功重定位帧的id，防止短时间多次重定位
        mnLastRelocFrameId = mCurrentFrame->mId;
        return true;
    }


}
template<typename T>
cv::Mat compute_dark_channel(const cv::Mat& origin, const int H, const int W, const int radius=7) {
    const T* const ori_ptr = origin.ptr<T>();
    // 决定数据类型, 是 uchar 还是 float
    const int TYPE = origin.type() == CV_8UC3 ? CV_8UC1 : CV_32FC1;
    // 【1】先 R, G, B 三通道求一个最小图
    cv::Mat min_bgr(H, W, TYPE);
    T* const min_bgr_ptr = min_bgr.ptr<T>();
    const int length = H * W;
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        min_bgr_ptr[i] = std::min(ori_ptr[p], std::min(ori_ptr[p + 1], ori_ptr[p + 2]));
    }
    // 【2】min_bgr 中每个点, 在窗口中找一个最小值
    // 先对图像做 padding
    cv::Mat padded_image;
    cv::copyMakeBorder(min_bgr, padded_image, radius, radius, radius, radius, cv::BORDER_REFLECT);
    auto pad_min_bgr = padded_image;
    const int H2 = H + 2 * radius;
    const int W2 = W + 2 * radius;
    // 存放第一次横向找最小值地结果
    cv::Mat temp(H2, W, TYPE);
    T* const temp_ptr = temp.ptr<T>();
    int cnt = 0;
    // 第一次, 横向找 H2 次
    for(int i = 0;i < H2; ++i) {
        T* const row_ptr = pad_min_bgr.ptr<T>() + i * W2 + radius;
        for(int j = 0;j < W; ++j) {
            T min_value = 255;
            for(int k = -radius; k <= radius; ++k)
                min_value = std::min(min_value, row_ptr[j + k]);
            temp_ptr[cnt++] = min_value;
        }
    }
    // 释放空间
    pad_min_bgr.release();
    // 第二次, 竖向比较
    for(int j = 0;j < W; ++j) {
        for(int i = 0;i < H; ++i) {
            T min_value = 255;
            const int offset = (radius + i) * W + j;
            for(int k = -radius; k <= radius; ++k)
                min_value = std::min(min_value, temp_ptr[offset + k * W]);
            min_bgr_ptr[i * W + j] = min_value;  // 结果直接存放到 min_bgr 里面, 节约空间
        }
    }
    return min_bgr;
}
std::list< std::pair<std::string, cv::Mat> > Tracking::dong_enhance(
        const cv::Mat& low_light,
        const int radius,
        const int A_pixels,
        const float weight,
        const float border,
        const bool denoise){
            std::list< std::pair<std::string, cv::Mat> > collections;
    // 获取信息
    const int H = low_light.rows;
    const int W = low_light.cols;
    const int C = low_light.channels();
    const int length = H * W;
    const int total_length = length * C;
    assert(low_light.type() == 16 and "Only CV_8U3(BGR) are supported !");

    //【1】首先 255 - low_light, 得到反转图像
    cv::Mat inverse(H, W, low_light.type());
    uchar* const inv_ptr = inverse.ptr<uchar>();
    for(int i = 0;i < total_length; ++i)
        inv_ptr[i] = 255 - low_light.data[i];
    collections.emplace_back("inverse", inverse);

    // 【2】对 inverse 去雾, 首先求 inverse 的暗通道
    const auto dark_channel = compute_dark_channel<uchar>(inverse, H, W, radius);
    collections.emplace_back("dark_channel", dark_channel);

    // 【3】根据暗通道, 排序筛选最大值的前 100 个点
    // 在 有雾图像中去找 rgb 之和最大值作为对三个通道大气光 A 的估计
    std::vector< std::vector<int> > book(256);
    const uchar* const dark_ptr = dark_channel.ptr<uchar>();
    for(int i = 0;i < length; ++i)
        book[dark_ptr[i]].emplace_back(i);
    int cnt = 0;
    std::vector<int> index(A_pixels);  // 找到暗通道最大的 100 个点坐标
    for(int i = 255; i >= 0; --i) {
        const int _size = book[i].size(); // 这里是 O(1)
        for(int t = 0;t < _size and cnt < A_pixels; ++t)
            index[cnt++] = book[i][t];
    }
    int max_bgr_sum = 0;  // 从这 100 个暗通道最大值对应 有雾图像 中, 找 r, g, b 之和最大的点
    int max_index = -1;
    for(int i = 0;i < A_pixels; ++i) {
        const int p = 3 * index[i];
        const int cur_sum = inv_ptr[p] + inv_ptr[p + 1] + inv_ptr[p + 2];
        if(cur_sum > max_bgr_sum) {
            max_bgr_sum = cur_sum;
            max_index = index[i];
        }
    }
    // 现在 max_index 是最大的 r, g, b 的通道
    int A[3] = {inv_ptr[max_index * 3], inv_ptr[max_index * 3 + 1], inv_ptr[max_index * 3 + 2]};
    //std::cout << "三个通道的全局大气光估计值 " << A[0] << ", " << A[1] << ", " << A[2] << "\n";

    // 【4】估计 t
    // 准备一个 float 图, 每个通道的像素除以对应的全局大气光
    cv::Mat R_A(H, W, CV_32FC3);
    float* const R_A_ptr = R_A.ptr<float>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        for(int c = 0;c < 3; ++c)
            R_A_ptr[p + c] = inv_ptr[p + c] * 1.f / A[c];
    }
    // 根据比值的暗通道, 得到透射率(这里的 R_A 是 CV_32FC1 了, 节约空间)
    R_A = compute_dark_channel<float>(R_A, H, W, radius);
    float* t_ptr = R_A.ptr<float>();

    // 对透射率做点小改变, 远景增强地更厉害点
    auto discount = [border](const float x) ->float {
        return x;
        if(x >= 0 and x < border) return 2 * x * x;
        else return x;
    };
    for(int i = 0;i < length; ++i)
        t_ptr[i] = discount(1.f - weight * t_ptr[i]);

    // 开始求解 J(x), 然后取反
    cv::Mat result(H, W, CV_8UC3);
    uchar* const res_ptr = result.ptr<uchar>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        for(int c = 0;c < 3; ++c) {
            const float J = (inv_ptr[p + c] - A[c]) * 1.f / t_ptr[i] + A[c];  // 偏大
            res_ptr[p + c] = cv::saturate_cast<uchar>(J);
        }
    }
    // 做去噪
    if(denoise) {
        cv::fastNlMeansDenoisingColored(result, result, 5, 5, 5, 15);
    }
    collections.emplace_back("dehazed", result.clone());

    // 反转图像
    for(int i = 0;i < total_length; ++i)
        res_ptr[i] = 255 - res_ptr[i]; // 更大
    collections.emplace_back("enhanced", result.clone());
    return collections;
        }
} //end ns
