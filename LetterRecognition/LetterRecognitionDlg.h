
// LetterRecognitionDlg.h : header file
//

#pragma once

#include "NeuralNetwork.h"

// CLetterRecognitionDlg dialog
class CLetterRecognitionDlg : public CDialogEx
{
// Construction
public:
	CLetterRecognitionDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_LETTERRECOGNITION_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnTrainFile();

public:
	int m_iMaxEpochs;
	double m_dTestSSE;
	double m_dTrainSSE;
	double m_dTrainPercent;
	double m_dLearningRate;
	double m_dTestPercent;
	CString m_sTest;

	CString m_sTestFile;
	CString m_sTrainFile;
private:
	NeuralNetwork m_nn;

	CSpinButtonCtrl m_spinEpochs;
	CSpinButtonCtrl m_spinRate;
	CEdit m_editEpochs;
	CEdit m_editRate;
public:
	afx_msg void OnDeltaposSpinEpochs(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnDeltaposSpinRate(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnBnClickedBtnTestFile();
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedBtnTest();
	afx_msg void OnBnClickedBtnTrainNn();
	CButton m_chkPretrain;
	CEdit m_editPretrain;
	int m_epochsPretrain;
	afx_msg void OnBnClickedCheckPretrain();
	CButton m_chkLastLayer;
	afx_msg void OnBnClickedBtnSaveWeights();
	afx_msg void OnBnClickedBtnLoadWeights();
	CString m_loadWFile;
	CString m_saveWFile;
	CComboBox m_comboFunc;
	afx_msg void OnBnClickedBtnShuffle();
	afx_msg void OnBnClickedBtnAssess();
};
