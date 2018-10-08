
import pandas as pd 
import numpy as np
import math
from scipy.stats import t as t_dist
from scipy.stats import f as f_dist
from numpy.linalg import matrix_rank
#   Defult values
X_COL=5


# Some Nessary Functions

def TestFullRank(Matrix_T): #Test Whether the Matrix is Full Rank
    if(matrix_rank(Matrix_T)==min(Matrix_T.shape)):
        return True
    else:
        print('The Matrix T is not a full rank Matrix')
        return False


# Data Input
Data=pd.read_excel('nerlove.xls',encoding='utf-8')
Data.describe()
for col in Data.columns:
    Data[col+'_L']=(Data[col].apply(math.log))
#print(Data.describe())
Xvar=['Q','PL','PF','PK','CONST']
Y=Data.iloc[:,-X_COL]
X=Data.iloc[:,1-X_COL:]
X['Const']=1
X_mat=np.mat(X)
Y_mat=np.mat(Y).T

#Caculate OLS estimator

b_coef=(X_mat.T * X_mat).I *(X_mat.T)*Y_mat #
Y_est=X_mat*b_coef
Y_res=Y_mat-Y_est #Residual
Y_mean=np.mean(Y_mat)

P_mat=X_mat*(X_mat.T*X_mat).I*X_mat.T
N_col=X_mat.shape[0]
K_col=X_mat.shape[1]
M_mat=np.mat(np.eye(N_col))-P_mat
SSR=Y_res.T*Y_res
S_square=SSR/(N_col-K_col)
R_UC=(Y_est.T*Y_est)/(Y_mat.T*Y_mat)
R_Cen=((Y_est-Y_mean).T *(Y_est-Y_mean)) /((Y_mat-Y_mean).T *(Y_mat-Y_mean))
Varb=np.float(S_square)*(X_mat.T*X_mat).I

Se_Bk=[]  #Standard Error of bk
t_value=[]
p_value=[]
Interval_Low=[]
Interval_High=[]
a_sig=0.025  #significance level
coef_list=[]
for i in range(len(Varb)):
    coef_list.append(float(b_coef[i]))
    Se_Bk.append(math.sqrt(Varb[i,i]))
    t_value.append(float(b_coef[i]/Se_Bk[i]))
    p_value.append(float(2*t_dist.sf(t_value[i],N_col-K_col,loc=0,scale=1)))
    Interval_High.append(float(t_dist.ppf(1-a_sig, N_col-K_col, loc=b_coef[i], scale=Se_Bk[i])))
    Interval_Low.append(float(t_dist.ppf(a_sig, N_col-K_col, loc=b_coef[i], scale=Se_Bk[i])))
    
Summary=pd.DataFrame({'Xvar':Xvar,'Coef':coef_list,'StdErr':Se_Bk,'t':t_value,'p_v':p_value,'L_Interval':Interval_Low,'H_Interval':Interval_High })

#Linear Hypothesis
R_mat=np.mat([[0,1,-1,0,0],[0,0,0,0,1]])
r_mat=np.mat([[0],[0]])
r_col=r_mat.shape[0]


F_value=(R_mat*b_coef-r_mat).T*(R_mat*Varb*R_mat.T).I*(R_mat*b_coef-r_mat) /r_col
P_Fvalue=f_dist.cdf(F_value,r_mat.shape[0],N_col-K_col)

#Restriced Linear Sqaure
Lambda=(R_mat*(X_mat.T*X_mat).I*R_mat.T).I*(r_mat-R_mat*b_coef)
br_coef=b_coef+(X_mat.T*X_mat).I*R_mat.T*Lambda
U_res=Y_mat-X_mat*br_coef
SSR_r=U_res.T*U_res
F_rvalue=((SSR_r-SSR)/(r_col))/(SSR/(N_col-K_col))
