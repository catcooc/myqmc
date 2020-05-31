using LinearAlgebra
dim=1
t=1
L=6^dim
U=4
#print(L)
mu=U/2
dt=0.125
beta=2
M=Int64(beta/dt)
jsl=0
gcs=4
bc=2
global jsl
global gcs
#print(M)
#M=2
lambda=acosh(exp(U*dt/2))
function fj(x,w)
     return([ceil(x/w) mod(x,w) == 0 ? w : mod(x,w)])
end

function ax(i)
       [i-x > 0 ? i-x : M-x+i  for x in 1:M]
end

function pd(x)
                if x <=0.5
                  return -1
                else 
                  return 1
                 end
end
function bj(x::Array,w::Int64)
		    map([ x + [1 0] ,x + [0 1] ,x + [-1 0],x + [0 -1]]) do z
	    	if findmin(z)[1] == 0
	    		z[findmin(z)[2]] = w
	    	elseif findmax(z)[1] == w+1
	    		z[findmax(z)[2]] = 1
	        else 
	        	z
             end
             return z
         end
end
function k(tz,Lz,dim =1,bh = -)
	if dim == 1 
		return [ (abs(bh(i,j))==1 || abs(bh(i,j)+Lz)==1) ? -tz : 0 for i=1:Lz , j=1:Lz ]
	else
	#return Matrix{Float64}(undef,mapreduce(x->fj(j,Int64(Lz^0.5)) == x,|,bj(fj(i,Int64(Lz^0.5)),Int64(Lz^0.5))) ? -tz : 0 for i=1:Lz , j=1:Lz )
	 	return [ mapreduce(x->fj(j,Int64(Lz^0.5)) == x,|,bj(fj(i,Int64(Lz^0.5)),Int64(Lz^0.5))) ? -tz : 0 for i=1:Lz , j=1:Lz ]
	end
end
#println(bj([1 1],4))
#print(k(-1.0,4))
#println()
#println(fj(4,4))
#print(exp(k(-0.5,4))-([0 0 0 1;0 0 1 0; 0 1 0 0;1 0 0 0]*sinh(1/2)+I*cosh(1/2)))
#print(exp(map(Float64,k(-1.0,4))))
#print(M)

#print(K)
 
function mmm(matr,arr,st=1)
	#print(matr[:,:,1])
	#print(arr[1],arr[2])
	mm=matr[:,:,arr[1]]

	for j in 1:(st-1)
		
		mm=matr[:,:,arr[1+j]]*mm

    end
	Uc,S,V=svd(mm)
	V=transpose(V)

    for i in st:st:length(arr)-st+1
    	
    	#println()
    	mm=matr[:,:,arr[i]]
    	for j in 1:(st-1)
            #println(j+i)
    		mm=matr[:,:,arr[i+j]]*mm
    	    
    	end
    		Uc,S,Vd=svd(mm*Uc*Diagonal(S))
            V=transpose(Vd)*V
    end
         
          F=svd(inv(Uc)*inv(V)+Diagonal(S))

    return inv(transpose(F.V)*V)*Diagonal(F.S.^-1)*inv(F.U*Uc)
 end  



function start()
#global vup 
#global vdown
global bup
global bdown
global K
global sm
#vdown=zeros(L,L,M)
#vup=zeros(L,L,M)


sm=map(pd,rand(M,L))
K=exp(map(Float64,k(t,L,dim)))



green()

end



function green()
	global gup
    global gdown
	# global vup 
	# global vdown
	global bup
	global bdown
    global K
	global sm
	bup=zeros(L,L,M)
	bdown=zeros(L,L,M)
    gup=zeros(L,L,M)
	gdown=zeros(L,L,M)
    for i  in 1:M
    	bup[:,:,i]=K*exp.(Diagonal(sm[i,:])*(lambda/dt)+I*(mu+U/2).*dt)
    	bdown[:,:,i]=K*exp.(Diagonal(sm[i,:])*(-lambda/dt)+I*(mu+U/2).*dt)
		# vup[:,:,i]=Diagonal(sm[i,:])*(lambda/dt)+I*(mu+U/2)
		# vdown[:,:,i]=Diagonal(sm[i,:])*(-lambda/dt)+I*(mu+U/2)
  #   	bup[:,:,i]=K*exp.(vup[:,:,i].*dt)
  #   	bdown[:,:,i]=K*exp.(vdown[:,:,i].*dt)
	end

	# for i in 1:M
    
 #    	Aup=I
 #    	Adown=I
 #    	for j in ax(i)
 #    		Aup = Aup*bup[:,:,j]
 #    		Adown =Adown*bdown[:,:,j]
 #    	end
    
	# 	gup[:,:,i]=inv(I+Aup)
	# 	gdown[:,:,i]=inv(I+Adown)
	# end
	for i in 1:M
		gup[:,:,i]=mmm(bup,ax(i)[end:-1:1],bc)
		gdown[:,:,i]=mmm(bdown,ax(i)[end:-1:1],bc)
	end
end

function rjs(x,y)

	global rup
	global rdown
	rup=(exp(-2*lambda*sm[y,x])-1)

	rdown=(exp(2*lambda*sm[y,x])-1)
	
	up=1+(1-gup[x,x,y])*rup
	down=1+(1-gdown[x,x,y])*rdown
	#print("xxx")
	return up*down 
end
function fastup(pl,tt)
    for i in 1:L
    	for j in 1:L
    		if i != j 
    			gup[i,j,tt]=gup[i,j,tt]-((-gup[i,pl,tt]*rup*gup[pl,j,tt])/(1+rup*(1-gup[pl,pl,tt])))

    	    	gdown[i,j,tt]=gdown[i,j,tt]-((-gdown[i,pl,tt]*rdown*gdown[pl,j,tt])/(1+rdown*(1-gdown[pl,pl,tt])))
    		else
    			gup[i,j,tt]=gup[i,j,tt]-(((1-gup[i,pl,tt])*rup*gup[pl,j,tt])/(1+rup*(1-gup[pl,pl,tt])))

    	    	gdown[i,j,tt]=gdown[i,j,tt]-(((1-gdown[i,pl,tt])*rdown*gdown[pl,j,tt])/(1+rdown*(1-gdown[pl,pl,tt])))
            end
        end
     end
end


function warm(cs)
	start()
	global jsl
	global gcs
	#print(size(sm))

	for nn in 1:cs
		#print(nn)

		for tt in 1:M 
            #println("m")
			#print(tt)
			for sp in 1:L
				#println("ss")
				
				if  (sp == 1 ) && (tt != 1)
					#print(sp)
                    gup[:,:,tt]=bup[:,:,tt-1]*gup[:,:,tt-1]*inv(bup[:,:,tt-1])
                    gdown[:,:,tt]=bdown[:,:,tt-1]*gdown[:,:,tt-1]*inv(bdown[:,:,tt-1])
                
                end    
				if rjs(sp,tt) > rand()
					#println("accept")
					jsl+= 1
					    if sp == L || jsl%gcs == 0
							#println("green")
							sm[tt,sp] = -sm[tt,sp]
							green()
						else
							#println("fastup")
							fastup(sp,tt) 
							sm[tt,sp] =-sm[tt,sp]
							#vup[:,:,tt]=Diagonal(sm[tt,:])*(lambda/dt)+I*(mu+U/2)
							#vdown[:,:,tt]=Diagonal(sm[tt,:])*(-lambda/dt)+I*(mu+U/2)
    						bup[:,:,tt]=K*exp.(Diagonal(sm[tt,:])*(lambda/dt)+I*(mu+U/2).*dt)
    						bdown[:,:,tt]=K*exp.(Diagonal(sm[tt,:])*(-lambda/dt)+I*(mu+U/2).*dt)
                    	end
                else
                    	#println("no accept")	
			    	
                end
		    end
        end
    #println()
    #print(nn,"jsl:",jsl/(nn*M*L))
    end	
            


end
#@time start()
@time warm(1000)
print(jsl/(1000*M*L))

# function startcs()
# global vup 
# global vdown
# global bup
# global bdown

# global sm
# global gup
# global gdown



# gup=zeros(L,L,M)
# gdown=zeros(L,L,M)
# vdown=zeros(L,L,M)
# vup=zeros(L,L,M)
# bup=zeros(L,L,M)
# bdown=zeros(L,L,M)

# sm=map(pd,rand(M,L))

# 	for i  in 1:M
# 		vup[:,:,i]=Diagonal(sm[i,:])*(lambda/dt)+I*(mu+U/2)
# 		vdown[:,:,i]=Diagonal(sm[i,:])*(-lambda/dt)+I*(mu+U/2)
#     	bup[:,:,i]=K*exp.(vup[:,:,i].*dt)
#     	bdown[:,:,i]=K*exp.(vdown[:,:,i].*dt)
#     end
# 	#for i in 1:M
#     #for i in 1:1
#     	Aup=I
#     	#Adown=I
#     	for j in ax(1)
#     		Aup = Aup*bup[:,:,j]
#     		#Adown = bdown[:,:,j]*Adown
#     	end
    
# 		print(inv(I+Aup))
# 		#gdown[:,:,i]=inv(I+Adown)
# 	#end
# # print(gup[:,:,1])

# # println()
# # print(gup[:,:,end])
# println()	
# uc,sc,dc=mmm(bup,ax(1)[end:-1:1])
# uc=inv(uc)
# dc=inv(dc)
# print(dc*inv(uc*dc+sc)*uc)

# #print(U)
# # println()
# # # print(S)
# # # print(V)
# # print("xxx")
# # Uc,S,V=svd(bup[:,:,1])
# # Uc,S,Vd=svd(bup[:,:,2]*Uc*Diagonal(S))
# # V=transpose(Vd)*transpose(V)
# # print(Uc*Diagonal(S)*(V))
# # println()
# # print(bup[:,:,2]*bup[:,:,1])
# # println()
# # Ucx,Sx,Vx=svd(Aup)
# # print(Ucx*Diagonal(Sx)*(Vx))

# end
# #print(ax(1))
# #print(ax(2))
# @time startcs()

