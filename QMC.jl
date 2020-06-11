using LinearAlgebra
dim=1
t=1
L=6^dim
U=4
#print(L)
mu=U/2
dt=0.125
#beta=0.5
#M=Int64(beta/dt)

gcs=3
bc=1
r=0
global worry
global jsl
global gcs
jsl=0
worry=0
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
		return [ (abs(bh(i,j))==1 || abs(bh(i,j)+Lz)==1 || abs(bh(i,j)-Lz)==1 ) ? -tz : 0 for i=1:Lz , j=1:Lz ]
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
	#println("arr",arr)
	#println("st:",st)
	mm=matr[:,:,arr[1]]

	for j in 1:(st-1)
		
		mm=matr[:,:,arr[1+j]]*mm

    end
	Uc,S,V=svd(mm)
	V=transpose(V)

    for i in (1+st):st:(length(arr)-st+1)
    	
    	#println(i)
    	mm=matr[:,:,arr[i]]
    	for j in 1:(st-1)
            #println(j+i)
            #println("xxx")
    		mm=matr[:,:,arr[i+j]]*mm
    	    
    	end
    		Uc,S,Vd=svd(mm*Uc*Diagonal(S))
            V=transpose(Vd)*V
    end
         
          F=svd(inv(Uc)*inv(V)+Diagonal(S))

   return inv(transpose(F.V)*V)*Diagonal(F.S.^-1)*inv(Uc*F.U)
   #return inv(V)*inv(inv(Uc)*inv(V)+Diagonal(S))*inv(Uc)
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
K=exp(map(Float64,k(-dt*t,L,dim)))
#K=exp([0 1 0 0 0 1 ; 1 0 1 0 0 0; 0 1 0 1 0 0; 0 0 1 0 1 0; 0 0 0 1 0 1 ;1 0 0 0 1 0].*dt)
#print(map(Float64,k(-dt*t,L,dim)))
#print(K)

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
    	bup[:,:,i]=K*(Diagonal(map(exp,sm[i,:]*(-1)*lambda.-(-mu+U/2)*dt)))
    	bdown[:,:,i]=K*(Diagonal(map(exp,sm[i,:]*(1)*lambda.-(-mu+U/2)*dt)))
    	# bup[:,:,i]=K*exp.(Diagonal(map(Float64,sm[i,:]*(-1)*lambda.-(mu-U/2)*dt)))
    	# bdown[:,:,i]=K*exp.(Diagonal(map(Float64,sm[i,:]*lambda.-(mu-U/2)*dt)))
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
    global r
	global rup
	global rdown
	global worry
	global upp
	global down

	rup=(exp(2*lambda*sm[y,x])-1)

	rdown=(exp(-2*lambda*sm[y,x])-1)
	
	upp=1+(1-gup[x,x,y])*rup
	down=1+(1-gdown[x,x,y])*rdown
	#print("xxx")
	r+= upp * down
	if upp*down < 0
		println(upp*down)
		worry=-1
	end
	return upp*down 
end


function fastup(pl,tt)
    for i in 1:L   
    	for j in 1:L
    		if i != pl 
    			gup[i,j,tt]=gup[i,j,tt]-((-gup[i,pl,tt]*rup*gup[pl,j,tt])/upp)#(1+rup*(1-gup[pl,pl,tt])))

    	    	gdown[i,j,tt]=gdown[i,j,tt]-((-gdown[i,pl,tt]*rdown*gdown[pl,j,tt])/down)#(1+rdown*(1-gdown[pl,pl,tt])))
    		else
    			gup[i,j,tt]=gup[i,j,tt]-(((1-gup[i,pl,tt])*rup*gup[pl,j,tt])/upp)#(1+rup*(1-gup[pl,pl,tt])))

    	    	gdown[i,j,tt]=gdown[i,j,tt]-(((1-gdown[i,pl,tt])*rdown*gdown[pl,j,tt])/down) #(1+rdown*(1-gdown[pl,pl,tt])))
            end
        end
     end
end


function up(sp,tt)
            global jsl

				if  (sp == 1 ) && (tt != 1)
					#print(sp)
					#for TT in  tt:M

                    gup[:,:,tt]=bup[:,:,tt-1]*gup[:,:,tt-1]*inv(bup[:,:,tt-1])
                    gdown[:,:,tt]=bdown[:,:,tt-1]*gdown[:,:,tt-1]*inv(bdown[:,:,tt-1])
                
                end    
				if rjs(sp,tt) > rand()
					#println("accept")
					accept=1
					jsl+= 1
					    if sp == L || jsl%gcs == 0
							#println("green")
							sm[tt,sp] = -sm[tt,sp]
							green()
						else
							#println("fastup")
							fastup(sp,tt) 
							sm[tt,sp] = - sm[tt,sp]
							#vup[:,:,tt]=Diagonal(sm[tt,:])*(lambda/dt)+I*(mu+U/2)
							#vdown[:,:,tt]=Diagonal(sm[tt,:])*(-lambda/dt)+I*(mu+U/2)
    						# bup[:,:,tt]=K*exp.(Diagonal(map(Float64,sm[tt,:]*(-1)*lambda.-(mu-U/2)*dt)))
    						# bdown[:,:,tt]=K*exp.(Diagonal(map(Float64,sm[tt,:]*lambda.-(mu-U/2)*dt)))
    						bup[:,:,tt]=K*(Diagonal(map(exp,sm[tt,:]*(-1)*lambda.-(-mu+U/2)*dt)))
							bdown[:,:,tt]=K*(Diagonal(map(exp,sm[tt,:]*(1)*lambda.-(-mu+U/2)*dt)))
                    	end
                else
                    	accept=0
                    	#println("no accept")	
			    end
	return accept
end

function warm(cs,cl=0,ck=1)
	global jsl
	global gcs
	global r
	println("warm")
    start()

	for nn in 1:cs
		#print(nn)

		for tt in 1:M 
            #println("m")
			#print(tt)
			for sp in 1:L
				#println("ss")
                up(sp,tt)
                
		    end
        end
    #println()
    #println(nn)
    end	
    println()
    println("measure")
    szg=zeros(ck)
    sxg=zeros(ck)     
    for nnn in 1:ck
    	sz=0
   		sx=0
    	for nn in 1:Int64(cl/ck)
		#print(nn)

			for tt in 1:M 
            #println("m")
			#print(tt)
				for sp in 1:L
				#println("ss")
     #            	if rjs(sp,tt) > rand()
     #            		sm[tt,sp] = -sm[tt,sp]
					# 	green()
					# end 
                		up(sp,tt)
                
                		
		    	end
        	end
        	sz+= mapreduce(x -> x/L,+,[(1-gup[i,i,1])^2 + (1-gup[i,i,1])*gup[i,i,1] + (1-gdown[i,i,1])*gdown[i,i,1] - (1-gdown[i,i,1])*(1-gup[i,i,1])- (1-gdown[i,i,1])*(1-gup[i,i,1]) + (1-gdown[i,i,1])^2  for i in 1:L])
        	#sz+=(1-gup[1,1,1])^2 + (1-gup[1,1,1])*gup[1,1,1] + (1-gdown[1,1,1])*gdown[1,1,1] - (1-gdown[1,1,1])*(1-gup[1,1,1])- (1-gdown[1,1,1])*(1-gup[1,1,1]) + (1-gdown[1,1,1])^2 
            #sx+=(1-gup[1,1,1])*gdown[1,1,1] + (1-gdown[1,1,1])*gup[1,1,1]
            sx+=mapreduce(x -> x/L,+,(1-gup[i,i,1])*gdown[i,i,1] + (1-gdown[i,i,1])*gup[i,i,1] for i in 1:L)
    #println()
    	#println(nn)
        
    	end
    	szg[nnn]=sz/(cl/ck)
        sxg[nnn]=sx/(cl/ck)
    	  
	end
	return szg,sxg

end
nnnn=8000
clll=1000
ccc=5
#@time start()
wd=[0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 3 3.5 4]
gcs=L*0.5
println("wd,gcs,bc:",length(wd),' ',gcs,' ',bc)
jg1=zeros(length(wd))
jg2=zeros(length(wd))
for tttt in 1:length(wd)
	global jsl
	global r 
	global M 
	global beta
	beta= wd[tttt]
	M=Int64(beta/dt)

	jsl=0
	r=0
println("beta M:",beta," ",M)
@time szz,sxx=warm(nnnn,clll,ccc)

jg1[tttt]=mapreduce(x -> x/ccc,+,szz)
jg2[tttt]=mapreduce(x -> x /ccc,+,sxx)
println("sz:",jg1[tttt])
println("sx:",jg2[tttt])
println("szwc:",sqrt(abs(mapreduce(x -> (x^2)/ccc,+,szz)-jg1[tttt])/ccc))
println("sxwc:",sqrt(abs(mapreduce(x -> (x^2)/ccc,+,sxx)-jg2[tttt])/ccc))
println("jsl:",jsl/(nnnn*M*L+clll*M*L))
println("r:",r/(nnnn*M*L+clll*M*L))
println("worry:",worry)
println()
end
println("z:", jg1)
println("x:",jg2)
#print(gup)
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
# function rjs(x,y)
#     global r
# 	global rup
# 	global rdown
# 	global worry
# 	rup=(exp(2*lambda*sm[y,x])-1)

# 	rdown=(exp(-2*lambda*sm[y,x])-1)
	
# 	up=1+(1-gup[x,x,y])*rup
# 	down=1+(1-gdown[x,x,y])*rdown
# 	#print("xxx")
# 	r+= up * down
# 	if up*down < 0
# 		println(up*down)
# 		worry=-1
# 	end
# 	return up*down 
# end
# function startcs()
# #global vup 
# #global vdown
# global bup
# global bdown
# global K
# global sm
# #vdown=zeros(L,L,M)
# #vup=zeros(L,L,M)


# sm=map(pd,rand(M,L))
# K=exp(map(Float64,k(-dt*t,L,dim)))
# #K=exp([0 1 0 0 0 1 ; 1 0 1 0 0 0; 0 1 0 1 0 0; 0 0 1 0 1 0; 0 0 0 1 0 1 ;1 0 0 0 1 0].*dt)
# #print(map(Float64,k(-dt*t,L,dim)))
# #print(sm)

# 	global gup
#     global gdown
# 	# global vup 
# 	# global vdown
# 	global bup
# 	global bdown
#     global K
# 	global sm
# 	bup=zeros(L,L,M)
# 	bdown=zeros(L,L,M)
#     gup=zeros(L,L,M)
# 	gdown=zeros(L,L,M)
# 	gupcs=zeros(L,L,M)
# 	gdowncs=zeros(L,L,M)
#     for i  in 1:M
    	
#     	bup[:,:,i]=K*(Diagonal(map(exp,sm[i,:]*(-1)*lambda.-(-mu+U/2)*dt)))
#     	bdown[:,:,i]=K*(Diagonal(map(exp,sm[i,:]*(1)*lambda.-(-mu+U/2)*dt)))
# 		# vup[:,:,i]=Diagonal(sm[i,:])*(lambda/dt)+I*(mu+U/2)
# 		# vdown[:,:,i]=Diagonal(sm[i,:])*(-lambda/dt)+I*(mu+U/2)
#   #   	bup[:,:,i]=K*exp.(vup[:,:,i].*dt)
#   #   	bdown[:,:,i]=K*exp.(vdown[:,:,i].*dt)
# 	end
#     #println("bup:",bup)
# 	for i in 1:M
    
#     	Aup=I
#     	Adown=I
#     	for j in ax(i)
#     		Aup = Aup*bup[:,:,j]
#     		Adown =Adown*bdown[:,:,j]
#     	end
    
# 		gupcs[:,:,i]=inv(I+Aup)
# 		gdowncs[:,:,i]=inv(I+Adown)
# 	end
# 	#println("gupcs:",gupcs)
# 	for i in 1:M
# 		# println(ax(i)[end:-1:1])
# 		gup[:,:,i]=mmm(bup,ax(i)[end:-1:1],bc)
# 		gdown[:,:,i]=mmm(bdown,ax(i)[end:-1:1],bc)
# 	end #startcs
    # println("up:",gup[:,:,1]-gupcs[:,:,1])
    # println("down:",gdown[:,:,1] - gdowncs[:,:,1])
    # Uc,S,V=svd(bup[:,:,1])
    # Uc,S,Vd=svd(bup[:,:,2]*Uc*Diagonal(S))
    # V=transpose(Vd)*transpose(V)
    # F=svd(inv(Uc)*inv(V)+Diagonal(S))

    # gcs=inv(transpose(F.V)*V)*Diagonal(F.S.^-1)*inv(Uc*F.U)
    #gcs=inv(V)*inv(inv(Uc)*inv(V)+Diagonal(S))*inv(Uc)
    # println("1:",det(gup[:,:,1]))
    # println("2:",det(gup[:,:,2]))
    #println("gupcs:", gupcs[:,:,1] - gcs)
# 			for tt in 1:M 
#             #println("m")
# 			#print(tt)
# 				for sp in 1:L
#                  rjs(sp,tt)
#              	end
# 			end
# 	return worry
# end

 # for nn in 1:1
	# worry=0
	
	# println("worry:",startcs())
 # end
