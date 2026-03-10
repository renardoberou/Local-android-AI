package com.localllm.app

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.localllm.app.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val vm: MainViewModel by viewModels()
    private val adapter = ChatAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        // RecyclerView
        binding.recycler.adapter = adapter
        binding.recycler.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }

        // Send button
        binding.btnSend.setOnClickListener {
            val text = binding.etInput.text.toString().trim()
            if (text.isNotEmpty()) {
                vm.send(text)
                binding.etInput.setText("")
            }
        }

        // Stop button
        binding.btnStop.setOnClickListener { vm.stopGeneration() }

        // Observe messages
        lifecycleScope.launch {
            vm.messages.collect { msgs ->
                adapter.submitList(msgs.toList()) {
                    binding.recycler.scrollToPosition(adapter.itemCount - 1)
                }
            }
        }

        // Observe state
        lifecycleScope.launch {
            vm.state.collect { state ->
                updateUi(state)
            }
        }

        // Kick off model init
        vm.initModel()
    }

    private fun updateUi(state: AppState) {
        when (state) {
            is AppState.Idle -> {
                binding.statusCard.visibility = View.VISIBLE
                binding.statusText.text = "Initialising…"
                binding.progressBar.visibility = View.GONE
                setInputEnabled(false)
            }
            is AppState.Downloading -> {
                binding.statusCard.visibility = View.VISIBLE
                val p = state.progress
                binding.statusText.text = if (p.totalBytes > 0) {
                    val mb = p.bytesDownloaded / 1_048_576
                    val total = p.totalBytes / 1_048_576
                    "Downloading model… ${mb} MB / ${total} MB (${p.percent}%)"
                } else {
                    "Starting download…"
                }
                binding.progressBar.visibility = View.VISIBLE
                binding.progressBar.progress = p.percent
                setInputEnabled(false)
            }
            is AppState.Loading -> {
                binding.statusCard.visibility = View.VISIBLE
                binding.statusText.text = "Loading model into memory…"
                binding.progressBar.visibility = View.INVISIBLE
                setInputEnabled(false)
            }
            is AppState.Ready -> {
                binding.statusCard.visibility = View.GONE
                binding.btnStop.visibility = View.GONE
                setInputEnabled(true)
            }
            is AppState.Generating -> {
                binding.statusCard.visibility = View.GONE
                binding.btnStop.visibility = View.VISIBLE
                setInputEnabled(false)
            }
            is AppState.Error -> {
                binding.statusCard.visibility = View.VISIBLE
                binding.statusText.text = "Error: ${state.msg}"
                binding.progressBar.visibility = View.GONE
                setInputEnabled(false)
                Toast.makeText(this, state.msg, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setInputEnabled(enabled: Boolean) {
        binding.etInput.isEnabled = enabled
        binding.btnSend.isEnabled = enabled
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_clear -> {
                vm.clearChat()
                true
            }
            R.id.action_info -> {
                AlertDialog.Builder(this)
                    .setTitle("Model Info")
                    .setMessage(vm.modelInfo())
                    .setPositiveButton("OK", null)
                    .show()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
