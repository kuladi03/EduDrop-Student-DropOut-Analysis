const express = require('express')
const app = express()



app.get('/' , async (req,res) => {
    res.json({
        msg:"Hi there"
    })
})

app.post('/dropout-analysis' , async (req,res) => {
    
})

app.listen(3000)