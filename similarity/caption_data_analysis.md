# **MSCOCO Cap-Similarity**

> ## **what is cap-similarity?**
* It is the similarity between different captions in one image.

</br>
</br>

>## **Train Dataset**
### **Similarity Distribution**  
<img src='train_sim_dist.png' width = 500 height = 300 >  

it's seems like nomal-distribution
</br>
</br>


### **The least similar example**  
<img src='288.png' width = 440 height = 300 >

* 'A man and a woman traveling down a road in front of building.'
* "A woman is riding down a hill on a child's toy"
* 'some dude tryng to score outside the trap house'
* 'two males and a female and one is looking out a window'
* 'Three people in a neighborhood conversing about something.'

    ### **Similarity:  0.159481** (img_id: 288651)  


### **The most similar example** 
<img src='431.png' width = 500 height = 300 >

* 'A black and white cat sitting in a chair. '
* 'A black and white cat sits on a brown chair.' 
* 'a black and white cat seated on the chair' 
* 'A black and white cat sitting on a chair.'
* 'A black and white cat sitting on a chair.'

    ### **Similarity:  0.960225** (img_id: 431756)
</br>
</br>
</br>
</br>
  
  
> ## **Validation Dataset**
### **Similarity Distribution** 
<img src='val_sim_dist.png' width = 500 height = 300 >
</br>
</br>

### **The least similar example**  
<img src='139.png' width = 500 height = 300 >

* 'The animal food tray and cage is empty.'
* 'some kind of cage that is next to a tree' 
* 'a couple of frisbees are stuck in a small tower' 
* 'A view of a piece of art near a tree.'
* 'A Frisbee golf net in a park with several Frisbee in it.'

    ### **Similarity: 0.182616** (img_id: 139883)

### **The most similar example**  
<img src='469.png' width = 500 height = 300 >

* 'Someone using a cell phone to take a picture of a baby.'    
* 'someone taking a picture of a baby using a cell phone.'   
* 'A person taking a picture of a baby on a cell phone.'  
* 'A person uses their cellphone to take a picture of a baby'   
* 'a person taking a photo of a baby with their cell phone '

    ### **Similarity: 0.969442** (img_id: 469398)

