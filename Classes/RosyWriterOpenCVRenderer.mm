
/*
 File: RosyWriterOpenCVRenderer.mm
 Abstract: n/a
 Version: 2.1
 
 Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple
 Inc. ("Apple") in consideration of your agreement to the following
 terms, and your use, installation, modification or redistribution of
 this Apple software constitutes acceptance of these terms.  If you do
 not agree with these terms, please do not use, install, modify or
 redistribute this Apple software.
 
 In consideration of your agreement to abide by the following terms, and
 subject to these terms, Apple grants you a personal, non-exclusive
 license, under Apple's copyrights in this original Apple software (the
 "Apple Software"), to use, reproduce, modify and redistribute the Apple
 Software, with or without modifications, in source and/or binary forms;
 provided that if you redistribute the Apple Software in its entirety and
 without modifications, you must retain this notice and the following
 text and disclaimers in all such redistributions of the Apple Software.
 Neither the name, trademarks, service marks or logos of Apple Inc. may
 be used to endorse or promote products derived from the Apple Software
 without specific prior written permission from Apple.  Except as
 expressly stated in this notice, no other rights or licenses, express or
 implied, are granted by Apple herein, including but not limited to any
 patent rights that may be infringed by your derivative works or by other
 works in which the Apple Software may be incorporated.
 
 The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
 MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
 THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
 OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
 
 IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
 MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
 AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
 STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
 
 Copyright (C) 2014 Apple Inc. All Rights Reserved.
 
 */

#import "RosyWriterOpenCVRenderer.h"

// To build OpenCV into the project:
//	- Download opencv2.framework for iOS
//	- Insert framework into project's Frameworks group
//	- Make sure framework is included under the target's Build Phases -> Link Binary With Libraries.
#import <opencv2/opencv.hpp>

@implementation RosyWriterOpenCVRenderer

#pragma mark RosyWriterRenderer

- (BOOL)operatesInPlace
{
    return YES;
}

- (FourCharCode)inputPixelFormat
{
    return kCVPixelFormatType_32BGRA;
}

- (void)prepareForInputWithFormatDescription:(CMFormatDescriptionRef)inputFormatDescription outputRetainedBufferCountHint:(size_t)outputRetainedBufferCountHint
{
    // nothing to do, we are stateless
}

- (void)reset
{
    // nothing to do, we are stateless
}

- (CVPixelBufferRef)copyRenderedPixelBuffer:(CVPixelBufferRef)pixelBuffer
{
    CVPixelBufferLockBaseAddress( pixelBuffer, 0 );
    
    unsigned char *base = (unsigned char *)CVPixelBufferGetBaseAddress( pixelBuffer );
    size_t width = CVPixelBufferGetWidth( pixelBuffer );
    size_t height = CVPixelBufferGetHeight( pixelBuffer );
    size_t stride = CVPixelBufferGetBytesPerRow( pixelBuffer );
    size_t extendedWidth = stride / sizeof( uint32_t ); // each pixel is 4 bytes/32 bits
    
    // Since the OpenCV Mat is wrapping the CVPixelBuffer's pixel data, we must do all of our modifications while its base address is locked.
    // If we want to operate on the buffer later, we'll have to do an expensive deep copy of the pixel data, using memcpy or Mat::clone().
    
    // Use extendedWidth instead of width to account for possible row extensions (sometimes used for memory alignment).
    // We only need to work on columms from [0, width - 1] regardless.
    
    cv::Mat bgraImage = cv::Mat( (int)height, (int)extendedWidth, CV_8UC4, base );
    
    
#if DETECT_RECT
    debugSquares([self findSquaresInImage:bgraImage], bgraImage);
#else
    for ( uint32_t y = 0; y < height; y++ )
    {
        for ( uint32_t x = 0; x < width; x++ )
        {
            
            bgraImage.at<cv::Vec<uint8_t,4> >(y,x)[1] = 0;
            
        }
    }
#endif //DETECT_RECT
    
    CVPixelBufferUnlockBaseAddress( pixelBuffer, 0 );
    return (CVPixelBufferRef)CFRetain( pixelBuffer );
}



double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 ) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

- (std::vector<std::vector<cv::Point> >)findSquaresInImage:(cv::Mat)_image
{
    std::vector<std::vector<cv::Point> > squares;
    cv::Mat pyr, timg, grayFrame, gray0(_image.size(), CV_8U), gray;
    int thresh = 31, N = 3;
    cv::pyrDown(_image, pyr, cv::Size(_image.cols/2, _image.rows/2));
    cv::pyrUp(pyr, timg, _image.size());
    //Convert smoothed image to grayscale
    cv::cvtColor(timg, grayFrame, cv::COLOR_RGB2GRAY);
    
    cv::blur( grayFrame, grayFrame, cv::Size(3,3) );
    
    std::vector<std::vector<cv::Point> > contours;
    
    //if using RGB, process over all 3 channels. i.e. c<3
    for( int c = 0; c < 1; c++ ) {
        int ch[] = {c, 0};
        
        //if processing over RGB, use timg instead of grayFrame.
        mixChannels(&grayFrame, 1, &gray0, 1, ch, 1);
        for( int l = 0; l < N; l++ ) {
            if( l == 0 ) {
                cv::Canny(gray0, gray, 0, thresh, 3);
                cv::dilate(gray, gray, cv::Mat(), cv::Point(-1,-1));
            }
            else {
                gray = gray0 >= (l+1)*255/N;
            }
            //            cv::findContours(gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            cv::findContours(gray, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
            
            std::vector<cv::Point> approx;
            for( size_t i = 0; i < contours.size(); i++ )
            {
                cv::approxPolyDP(cv::Mat(contours[i]), approx, arcLength(cv::Mat(contours[i]), true)*0.02, true);
                if( approx.size() == 4 && fabs(contourArea(cv::Mat(approx))) > 1000 && cv::isContourConvex(cv::Mat(approx))) {
                    double maxCosine = 0;
                    
                    for( int j = 2; j < 5; j++ )
                    {
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    
                    if( maxCosine < 0.3 ) {
                        squares.push_back(approx);
                    }
                }
            }
        }
    }
    return squares;
}

void debugSquares( std::vector<std::vector<cv::Point> > squares, cv::Mat image )
{
    
#if SHOW_GRAYFRAME
    cv::Mat grayFrame;
    cv::cvtColor(image, grayFrame, cv::COLOR_RGB2GRAY);
    for ( int i = 0; i< squares.size(); i++ ) {
        
        // draw bounding rect
        cv::Rect rect = boundingRect(cv::Mat(squares[i]));
        // cv::rectangle(grayFrame, rect.tl(), rect.br(), cv::Scalar(0,255,0), 2, 8, 0);
        
    }
    
    
    
    return grayFrame;
#else
    cv::Rect outsideRect;
    if(squares.size()>0){
        outsideRect = boundingRect(cv::Mat(squares[0]));
    }
    
    for ( int i = 1; i< squares.size(); i++ ) {
        
        //  NSLog([NSString stringWithFormat:@"%.0f %.0f, %.0f %.0f", squares])
        // draw contour
        // cv::drawContours(image, squares, i, cv::Scalar(255,0,0), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        
        // draw bounding rect
        cv::Rect rect = boundingRect(cv::Mat(squares[i]));
        
        //If multiple rectangles are found, calculate the one that bounds the other
        outsideRect = outsideRect | rect ;
        //        cv::rectangle(image, rect.tl(), rect.br(), cv::Scalar(0,255,0), 2, 8, 0);
        
        
        // draw rotated rect
        /*        cv::RotatedRect minRect = minAreaRect(cv::Mat(squares[i]));
         cv::Point2f rect_points[4];
         minRect.points( rect_points );
         for ( int j = 0; j < 4; j++ ) {
         cv::line( image, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0,0,255), 1, 8 ); // blue
         }*/
        
    }
    
    //Draw the rectangle on the original image
    cv::Mat tempImage;
    image.copyTo(tempImage);
    cv::rectangle(tempImage, outsideRect.tl(), outsideRect.br(), cv::Scalar(63,0,77), cv::FILLED, 8, 0);
    //Blend the rectangle with the original image to get the transparency
    cv::addWeighted(tempImage, 0.6, image, 1, 1, image);
    
    //Draw a thick outside border
     cv::rectangle(tempImage, outsideRect.tl(), outsideRect.br(), cv::Scalar(63,0,77), 10, 8, 0);
#endif //SHOW_GRAYFRAME
    
}






@end
